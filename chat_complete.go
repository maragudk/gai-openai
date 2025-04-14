package openai

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/openai/openai-go"
	"maragu.dev/gai"
)

type ChatCompleteModel string

const (
	ChatCompleteModelGPT4o     = ChatCompleteModel(openai.ChatModelGPT4o)
	ChatCompleteModelGPT4oMini = ChatCompleteModel(openai.ChatModelGPT4oMini)
)

type ChatCompleter struct {
	Client *openai.Client
	log    *slog.Logger
	model  ChatCompleteModel
}

type NewChatCompleterOptions struct {
	Model ChatCompleteModel
}

func (c *Client) NewChatCompleter(opts NewChatCompleterOptions) *ChatCompleter {
	return &ChatCompleter{
		Client: c.Client,
		log:    c.log,
		model:  opts.Model,
	}
}

// ChatComplete satisfies [gai.ChatCompleter].
func (c *ChatCompleter) ChatComplete(ctx context.Context, req gai.ChatCompleteRequest) (gai.ChatCompleteResponse, error) {
	var messages []openai.ChatCompletionMessageParamUnion

	for _, m := range req.Messages {
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
		Model:    openai.F(openai.ChatModel(c.model)),
	}

	if req.Temperature != nil {
		params.Temperature = openai.F(req.Temperature.Float64())
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

var _ gai.ChatCompleter = (*ChatCompleter)(nil)
