package openai

import (
	"context"
	"log/slog"

	"github.com/openai/openai-go"
	"maragu.dev/errors"
	"maragu.dev/gai"
)

type EmbedModel string

const (
	EmbedModelTextEmbedding3Large = EmbedModel(openai.EmbeddingModelTextEmbedding3Large)
	EmbedModelTextEmbedding3Small = EmbedModel(openai.EmbeddingModelTextEmbedding3Small)
)

type Embedder struct {
	Client     openai.Client
	dimensions int
	log        *slog.Logger
	model      EmbedModel
}

type NewEmbedderOptions struct {
	Dimensions int
	Model      EmbedModel
}

func (c *Client) NewEmbedder(opts NewEmbedderOptions) *Embedder {
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
func (e *Embedder) Embed(ctx context.Context, req gai.EmbedRequest) (gai.EmbedResponse[float64], error) {
	v := gai.ReadAllString(req.Input)

	res, err := e.Client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input:          openai.EmbeddingNewParamsInputUnion{OfString: openai.Opt(v)},
		Model:          openai.EmbeddingModel(e.model),
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
		Dimensions:     openai.Opt(int64(e.dimensions)),
	})
	if err != nil {
		return gai.EmbedResponse[float64]{}, errors.Wrap(err, "error embedding")
	}
	if len(res.Data) == 0 {
		return gai.EmbedResponse[float64]{}, errors.New("no embeddings returned")
	}

	return gai.EmbedResponse[float64]{
		Embedding: res.Data[0].Embedding,
	}, nil
}

var _ gai.Embedder[float64] = (*Embedder)(nil)
