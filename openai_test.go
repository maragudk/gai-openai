package openai_test

import (
	"testing"

	"maragu.dev/env"
	"maragu.dev/gai"
	"maragu.dev/is"

	"maragu.dev/openai"
)

func TestNewOpenAIClient(t *testing.T) {
	t.Run("can create a new client with a key", func(t *testing.T) {
		client := gai.NewOpenAIClient(gai.NewOpenAIClientOptions{Key: "123"})
		is.NotNil(t, client)
	})
}

func TestOpenAIClient_Complete(t *testing.T) {
	t.Run("can send a streaming chat completion request", func(t *testing.T) {
		c := newClient()

		prompt := gai.Prompt{
			Model: gai.ModelGPT4oMini,
			Messages: []gai.Message{
				gai.NewUserTextMessage("Hi!"),
			},
			Temperature: gai.Ptr(0.0),
		}

		res := c.Complete(t.Context(), prompt)

		var text string
		for part, err := range res.Parts() {
			is.NotError(t, err)
			text += part.Text()
		}
		is.Equal(t, "Hello! How can I assist you today?", text)
	})
}

func newClient() *openai.Client {
	_ = env.Load(".env.test.local")

	return openai.NewClient(openai.NewClientOptions{Key: env.GetStringOrDefault("OPENAI_KEY", "")})
}
