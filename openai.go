package openai

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"maragu.dev/errors"
	"maragu.dev/gai"
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

type ChatCompleter struct {
	Client *openai.Client
	log    *slog.Logger
	model  ChatCompleteModel
}

type ChatCompleteModel string

const (
	ChatCompleteModelGPT4o     = ChatCompleteModel(openai.ChatModelGPT4o)
	ChatCompleteModelGPT4oMini = ChatCompleteModel(openai.ChatModelGPT4oMini)
)

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

type Embedder struct {
	Client     *openai.Client
	dimensions int
	log        *slog.Logger
	model      EmbedModel
}

type EmbedModel string

const (
	EmbedModelTextEmbedding3Large = EmbedModel(openai.EmbeddingModelTextEmbedding3Large)
	EmbedModelTextEmbedding3Small = EmbedModel(openai.EmbeddingModelTextEmbedding3Small)
)

type NewEmbedderOptions struct {
	Dimensions int
	Model      EmbedModel
}

func (c *Client) NewEmbedder(opts NewEmbedderOptions) *Embedder {
	// Validate dimensions
	if opts.Dimensions <= 0 {
		panic("dimensions must be greater than 0")
	}

	switch opts.Model {
	case EmbedModelTextEmbedding3Large:
		if opts.Dimensions > 3072 {
			panic("dimensions must be less than or equal to 3072")
		}
	case EmbedModelTextEmbedding3Small:
		if opts.Dimensions > 1536 {
			panic("dimensions must be less than or equal to 1536")
		}
	}

	return &Embedder{
		Client:     c.Client,
		dimensions: opts.Dimensions,
		log:        c.log,
		model:      opts.Model,
	}
}

// Embed satisfies [gai.Embedder].
func (c *Embedder) Embed(ctx context.Context, req gai.EmbedRequest) (gai.EmbedResponse[float64], error) {
	v := gai.ReadAllString(req.Input)

	res, err := c.Client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input:          openai.F[openai.EmbeddingNewParamsInputUnion](shared.UnionString(v)),
		Model:          openai.F(openai.EmbeddingModel(c.model)),
		EncodingFormat: openai.F(openai.EmbeddingNewParamsEncodingFormatFloat),
		Dimensions:     openai.F(int64(c.dimensions)),
	})
	if err != nil {
		return gai.EmbedResponse[float64]{}, errors.Wrap(err, "error creating embeddings")
	}
	if len(res.Data) == 0 {
		return gai.EmbedResponse[float64]{}, errors.New("no embeddings returned")
	}

	return gai.EmbedResponse[float64]{
		Embedding: res.Data[0].Embedding,
	}, nil
}

var _ gai.Embedder[float64] = (*Embedder)(nil)
