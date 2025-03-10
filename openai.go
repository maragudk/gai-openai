package openai

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"maragu.dev/gai"
)

const (
	ModelGPT4o     = gai.ChatModel(openai.ChatModelGPT4o)
	ModelGPT4oMini = gai.ChatModel(openai.ChatModelGPT4oMini)
)

type Client struct {
	Client *openai.Client
	log    *slog.Logger
}

type NewClientOptions struct {
	BaseURL string
	Key     string
	Log     *slog.Logger
}

func NewClient(opts NewClientOptions) *Client {
	if opts.Log == nil {
		opts.Log = slog.New(slog.DiscardHandler)
	}

	var clientOpts []option.RequestOption

	if opts.BaseURL != "" {
		if !strings.HasSuffix(opts.BaseURL, "/") {
			opts.BaseURL += "/"
		}
		clientOpts = append(clientOpts, option.WithBaseURL(opts.BaseURL))
	}

	if opts.Key != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(opts.Key))
	}

	return &Client{
		Client: openai.NewClient(clientOpts...),
		log:    opts.Log,
	}
}

// ChatComplete satisfies [gai.ChatCompleter].
func (c *Client) ChatComplete(ctx context.Context, p gai.Prompt) (gai.ChatCompleteResponse, error) {
	var messages []openai.ChatCompletionMessageParamUnion

	for _, m := range p.Messages {
		switch m.Role {
		case gai.MessageRoleUser:
			var parts []openai.ChatCompletionContentPartUnionParam
			for _, part := range m.Parts {
				switch part.Type {
				case gai.MessagePartTypeText:
					parts = append(parts, openai.TextPart(part.Text()))
				default:
					panic("not implemented")
				}
			}
			messages = append(messages, openai.UserMessageParts(parts...))

		default:
			panic("not implemented")
		}
	}

	params := openai.ChatCompletionNewParams{
		Messages: openai.F(messages),
		Model:    openai.F(openai.ChatModel(p.Model)),
	}

	if p.Temperature != nil {
		params.Temperature = openai.F(*p.Temperature)
	}

	stream := c.Client.Chat.Completions.NewStreaming(ctx, params)

	return gai.NewChatCompleteResponse(func(yield func(gai.MessagePart, error) bool) {
		defer func() {
			if err := stream.Close(); err != nil {
				c.log.Info("Error closing stream", "error", err)
			}
		}()

		var acc openai.ChatCompletionAccumulator
		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)

			if _, ok := acc.JustFinishedContent(); ok {
				break
			}

			if _, ok := acc.JustFinishedToolCall(); ok {
				continue
				// TODO handle tool call
				// println("Tool call stream finished:", tool.Index, tool.Name, tool.Arguments)
			}

			if refusal, ok := acc.JustFinishedRefusal(); ok {
				yield(gai.MessagePart{}, fmt.Errorf("refusal: %v", refusal))
				return
			}

			if len(chunk.Choices) > 0 {
				if !yield(gai.TextMessagePart(chunk.Choices[0].Delta.Content), nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(gai.MessagePart{}, err)
		}
	}), nil
}

var _ gai.ChatCompleter = (*Client)(nil)
