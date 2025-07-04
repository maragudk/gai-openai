package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/openai/openai-go"
	"maragu.dev/gai"
)

type ChatCompleteModel string

const (
	ChatCompleteModelGPT4o     = ChatCompleteModel(openai.ChatModelGPT4o)
	ChatCompleteModelGPT4oMini = ChatCompleteModel(openai.ChatModelGPT4oMini)
)

type ChatCompleter struct {
	Client openai.Client
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

	if req.System != nil {
		messages = append(messages, openai.SystemMessage(*req.System))
	}

	for _, m := range req.Messages {
		switch m.Role {
		case gai.MessageRoleUser:
			var parts []openai.ChatCompletionContentPartUnionParam

			for _, part := range m.Parts {
				switch part.Type {
				case gai.MessagePartTypeText:
					parts = append(parts, openai.ChatCompletionContentPartUnionParam{
						OfText: &openai.ChatCompletionContentPartTextParam{Text: part.Text()},
					})

				case gai.MessagePartTypeToolResult:
					// Even though this is just a part, we append to messages directly

					// Take existing parts and append to messages first
					if len(parts) > 0 {
						messages = append(messages, openai.UserMessage(parts))
					}
					parts = nil

					toolResult := part.ToolResult()
					content := toolResult.Content
					if toolResult.Err != nil {
						content = fmt.Sprintf("Error: %s", toolResult.Err)
					}
					messages = append(messages, openai.ToolMessage(content, toolResult.ID))
					continue

				default:
					panic("not implemented")
				}
			}

			if len(parts) > 0 {
				messages = append(messages, openai.UserMessage(parts))
			}

		case gai.MessageRoleModel:
			var parts []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion

			for _, part := range m.Parts {
				switch part.Type {
				case gai.MessagePartTypeText:
					parts = append(parts, openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
						OfText: &openai.ChatCompletionContentPartTextParam{Text: part.Text()},
					})

				case gai.MessagePartTypeToolCall:
					// Even though this is just a part, we append to messages directly

					// Take existing parts and append to messages first
					if len(parts) > 0 {
						messages = append(messages, openai.AssistantMessage(parts))
					}
					parts = nil

					toolCall := part.ToolCall()
					messages = append(messages, openai.ChatCompletionMessageParamUnion{
						OfAssistant: &openai.ChatCompletionAssistantMessageParam{
							ToolCalls: []openai.ChatCompletionMessageToolCallParam{
								{
									ID: toolCall.ID,
									Function: openai.ChatCompletionMessageToolCallFunctionParam{
										Name:      toolCall.Name,
										Arguments: string(toolCall.Args),
									},
								},
							},
						},
					})
					continue

				default:
					panic("not implemented")
				}
			}

			if len(parts) > 0 {
				messages = append(messages, openai.AssistantMessage(parts))
			}

		default:
			panic("unknown role " + m.Role)
		}
	}

	var tools []openai.ChatCompletionToolParam
	for _, tool := range req.Tools {
		tools = append(tools, openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: openai.String(tool.Description),
				Parameters: openai.FunctionParameters{
					"type":       "object",
					"properties": normalizeToolSchemaProperties(tool.Schema.Properties),
				},
			},
		})
	}

	params := openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    openai.ChatModel(c.model),
		Tools:    tools,
	}

	if req.Temperature != nil {
		params.Temperature = openai.Opt(req.Temperature.Float64())
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

			if toolCall, ok := acc.JustFinishedToolCall(); ok {
				if !yield(gai.ToolCallPart(toolCall.ID, toolCall.Name, json.RawMessage(toolCall.Arguments)), nil) {
					return
				}
				continue
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

// normalizeToolSchemaProperties recursively normalizes schema properties for OpenAI compatibility
func normalizeToolSchemaProperties(properties map[string]*gai.Schema) map[string]*gai.Schema {
	if len(properties) == 0 {
		return properties
	}

	result := make(map[string]*gai.Schema)
	for key, schema := range properties {
		result[key] = normalizeToolSchema(schema)
	}
	return result
}

// normalizeToolSchema creates a normalized copy of a gai.Schema with lowercase type names
func normalizeToolSchema(schema *gai.Schema) *gai.Schema {
	if schema == nil {
		return nil
	}

	// Create a copy of the schema
	normalized := &gai.Schema{
		AnyOf:            schema.AnyOf,
		Default:          schema.Default,
		Description:      schema.Description,
		Enum:             schema.Enum,
		Example:          schema.Example,
		Format:           schema.Format,
		Items:            normalizeToolSchema(schema.Items),
		MaxItems:         schema.MaxItems,
		Maximum:          schema.Maximum,
		MinItems:         schema.MinItems,
		Minimum:          schema.Minimum,
		Properties:       normalizeToolSchemaProperties(schema.Properties),
		PropertyOrdering: schema.PropertyOrdering,
		Required:         schema.Required,
		Title:            schema.Title,
		Type:             gai.SchemaType(strings.ToLower(string(schema.Type))),
	}

	// Recursively normalize anyOf schemas
	if len(schema.AnyOf) > 0 {
		normalized.AnyOf = make([]*gai.Schema, len(schema.AnyOf))
		for i, s := range schema.AnyOf {
			normalized.AnyOf[i] = normalizeToolSchema(s)
		}
	}

	return normalized
}

var _ gai.ChatCompleter = (*ChatCompleter)(nil)
