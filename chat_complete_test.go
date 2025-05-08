package openai_test

import (
	"testing"

	"maragu.dev/gai"
	"maragu.dev/is"

	openai "maragu.dev/gai-openai"
)

func TestChatCompleter_ChatComplete(t *testing.T) {
	t.Run("can chat-complete", func(t *testing.T) {
		cc := newChatCompleter(t)

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

			switch part.Type {
			case gai.MessagePartTypeText:
				output += part.Text()

			default:
				t.Fatal("unexpected message parts")
			}
		}
		is.Equal(t, "Hello! How can I assist you today?", output)

		req.Messages = append(req.Messages, gai.NewModelTextMessage("Hello! How can I assist you today?"))
		req.Messages = append(req.Messages, gai.NewUserTextMessage("What does the acronym AI stand for? Be brief."))

		res, err = cc.ChatComplete(t.Context(), req)
		is.NotError(t, err)

		output = ""
		for part, err := range res.Parts() {
			is.NotError(t, err)

			switch part.Type {
			case gai.MessagePartTypeText:
				output += part.Text()

			default:
				t.Fatal("unexpected message parts")
			}
		}
		is.Equal(t, `AI stands for "Artificial Intelligence."`, output)
	})
}

func newChatCompleter(t *testing.T) *openai.ChatCompleter {
	c := newClient(t)
	cc := c.NewChatCompleter(openai.NewChatCompleterOptions{
		Model: openai.ChatCompleteModelGPT4oMini,
	})
	return cc
}
