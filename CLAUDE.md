# Development Guidelines

## Build/Test Commands
- Run all tests: `go test -coverprofile=cover.out -shuffle on ./...`
- Run single test: `go test -run TestName` (e.g. `go test -run TestClient_ChatComplete`)
- View test coverage: `go tool cover -html=cover.out`
- Run benchmarks: `go test -bench=. ./...`
- Run linter: `golangci-lint run`

## Code Style
- **Imports**: Standard lib first, third-party next, local last. Alphabetized within sections.
- **Naming**: PascalCase for exports, camelCase for privates, constants use `ModelPrefixName` format
- **Types**: Defined at top of file, verify interface compliance with `var _ InterfaceName = (*ConcreteType)(nil)`
- **Error Handling**: Propagate up call stack, handle explicitly, use `fmt.Errorf` for context
- **Functions**: Small with clear responsibilities, function options pattern for configuration
- **Tests**: Table-driven tests in separate `_test.go` files, use `t.Run` for subtests
- **Documentation**: Follow Go doc conventions, document interface implementations

The codebase implements OpenAI API client functionality compatible with the `gai` interface.
