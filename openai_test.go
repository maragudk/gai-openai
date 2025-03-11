package openai_test

import (
	"strings"
	"testing"

	"maragu.dev/env"
	"maragu.dev/gai"
	"maragu.dev/is"

	openai "maragu.dev/gai-openai"
)

func TestNewClient(t *testing.T) {
	t.Run("can create a new client with a key", func(t *testing.T) {
		client := openai.NewClient(openai.NewClientOptions{Key: "123"})
		is.NotNil(t, client)
	})
}

func TestClient_ChatComplete(t *testing.T) {
	t.Run("can send a streaming chat completion request", func(t *testing.T) {
		c := newClient()

		cc := c.NewChatCompleter(openai.NewChatCompleterOptions{Model: openai.ChatCompleteModelGPT4oMini})

		req := gai.ChatCompleteRequest{
			Messages: []gai.Message{
				gai.NewUserTextMessage("Hi!"),
			},
			Temperature: gai.Ptr(gai.Temperature(0)),
		}

		res, err := cc.ChatComplete(t.Context(), req)
		is.NotError(t, err)

		var output string
		for part, err := range res.Parts() {
			is.NotError(t, err)
			output += part.Text()
		}
		is.Equal(t, "Hello! How can I assist you today?", output)
	})
}

func TestClient_Embed(t *testing.T) {
	t.Run("can create a text embedding", func(t *testing.T) {
		c := newClient()

		e := c.NewEmbedder(openai.NewEmbedderOptions{
			Model:      openai.EmbedModelTextEmbedding3Small,
			Dimensions: 1536,
		})

		req := gai.EmbedRequest{
			Input: strings.NewReader("Embed this, please."),
		}

		res, err := e.Embed(t.Context(), req)
		is.NotError(t, err)

		is.Equal(t, 1536, len(res.Embedding))
	})
}

func newClient() *openai.Client {
	_ = env.Load(".env.test.local")

	return openai.NewClient(openai.NewClientOptions{Key: env.GetStringOrDefault("OPENAI_KEY", "")})
}
