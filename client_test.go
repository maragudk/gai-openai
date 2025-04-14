package openai_test

import (
	"testing"

	"maragu.dev/env"
	"maragu.dev/is"

	openai "maragu.dev/gai-openai"
)

func TestNewClient(t *testing.T) {
	t.Run("can create a new client with a key", func(t *testing.T) {
		client := openai.NewClient(openai.NewClientOptions{Key: "123"})
		is.NotNil(t, client)
	})
}

func newClient() *openai.Client {
	_ = env.Load(".env.test.local")

	return openai.NewClient(openai.NewClientOptions{Key: env.GetStringOrDefault("OPENAI_KEY", "")})
}
