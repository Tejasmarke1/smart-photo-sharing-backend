# Contributing to Kwikpic

Thank you for your interest in contributing!

## Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`poetry run pytest`)
5. Run linting (`poetry run black . && poetry run ruff check .`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Use Ruff for linting
- Add type hints to all functions
- Write docstrings for public APIs

## Testing

- Write tests for all new features
- Maintain test coverage above 80%
- Run full test suite before submitting PR

## Commit Messages

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/tooling changes
