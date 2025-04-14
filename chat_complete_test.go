package openai_test

import (
	"testing"

	"maragu.dev/gai"
	"maragu.dev/is"

	openai "maragu.dev/gai-openai"
)

func TestChatCompleter_ChatComplete(t *testing.T) {
	t.Run("can chat-complete", func(t *testing.T) {
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
