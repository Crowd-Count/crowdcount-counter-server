# Makefile - Build automation for Video Streaming Platform

.PHONY: help install build test deploy clean docker-build docker-run docker-stop logs

# Variables
PROJECT_NAME = video-streaming-platform
DOCKER_IMAGE = video-streaming-server
DOCKER_TAG = latest
SERVER_DIR = server
CLIENT_DIR = client
INFRA_DIR = infrastructure

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

# Default target
help:
	@echo "$(GREEN)Video Streaming Platform - Build Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make install       - Install all dependencies"
	@echo "  make dev          - Run development server"
	@echo "  make test         - Run all tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo ""
	@echo "$(YELLOW)Build:$(NC)"
	@echo "  make build        - Build all components"
	@echo "  make build-server - Build server only"
	@echo "  make build-client - Build client applications"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-run   - Run with Docker Compose"
	@echo "  make docker-stop  - Stop Docker containers"
	@echo "  make docker-clean - Remove Docker resources"
	@echo ""
	@echo "$(YELLOW)Deployment:$(NC)"
	@echo "  make deploy-dev   - Deploy to development"
	@echo "  make deploy-prod  - Deploy to production"
	@echo ""
	@echo "$(YELLOW)Utilities:$(NC)"
	@echo "  make logs         - Show server logs"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make setup        - Initial project setup"

# Development targets
install:
	@echo "$(GREEN)Installing dependencies...$(NC)"
	cd $(SERVER_DIR) && npm install
	@if [ -d "$(CLIENT_DIR)/sender" ]; then cd $(CLIENT_DIR)/sender && npm install; fi
	@if [ -d "$(CLIENT_DIR)/viewer" ]; then cd $(CLIENT_DIR)/viewer && npm install; fi
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

dev:
	@echo "$(GREEN)Starting development server...$(NC)"
	cd $(SERVER_DIR) && npm run dev

test:
	@echo "$(GREEN)Running tests...$(NC)"
	cd $(SERVER_DIR) && npm test

test-watch:
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	cd $(SERVER_DIR) && npm run test:watch

lint:
	@echo "$(GREEN)Running linter...$(NC)"
	cd $(SERVER_DIR) && npm run lint

format:
	@echo "$(GREEN)Formatting code...$(NC)"
	cd $(SERVER_DIR) && npm run format

# Build targets
build: build-server build-client
	@echo "$(GREEN)Build complete!$(NC)"

build-server:
	@echo "$(GREEN)Building server...$(NC)"
	cd $(SERVER_DIR) && npm run build 2>/dev/null || echo "No build script defined"

build-client:
	@echo "$(GREEN)Building client applications...$(NC)"
	@if [ -d "$(CLIENT_DIR)/sender" ]; then cd $(CLIENT_DIR)/sender && npm run build; fi
	@if [ -d "$(CLIENT_DIR)/viewer" ]; then cd $(CLIENT_DIR)/viewer && npm run build; fi

# Docker targets
docker-build:
	@echo "$(GREEN)Building Docker images...$(NC)"
	cd $(INFRA_DIR)/docker && docker-compose build

docker-run:
	@echo "$(GREEN)Starting Docker containers...$(NC)"
	cd $(INFRA_DIR)/docker && docker-compose up -d
	@echo "$(GREEN)Containers started! Server running at http://localhost:3000$(NC)"

docker-stop:
	@echo "$(YELLOW)Stopping Docker containers...$(NC)"
	cd $(INFRA_DIR)/docker && docker-compose down

docker-clean:
	@echo "$(RED)Removing Docker resources...$(NC)"
	cd $(INFRA_DIR)/docker && docker-compose down -v --rmi all

docker-logs:
	cd $(INFRA_DIR)/docker && docker-compose logs -f video-server

# Deployment targets
deploy-dev:
	@echo "$(GREEN)Deploying to development environment...$(NC)"
	cd $(INFRA_DIR)/terraform/environments/dev && terraform init && terraform apply

deploy-staging:
	@echo "$(YELLOW)Deploying to staging environment...$(NC)"
	cd $(INFRA_DIR)/terraform/environments/staging && terraform init && terraform apply

deploy-prod:
	@echo "$(RED)Deploying to production environment...$(NC)"
	@echo "$(YELLOW)Are you sure? [y/N]$(NC)"
	@read -r REPLY; \
	if [ "$$REPLY" = "y" ] || [ "$$REPLY" = "Y" ]; then \
		cd $(INFRA_DIR)/terraform/environments/production && terraform init && terraform apply; \
	else \
		echo "$(YELLOW)Deployment cancelled$(NC)"; \
	fi

# Utility targets
logs:
	@echo "$(GREEN)Showing server logs...$(NC)"
	@if [ -f "$(SERVER_DIR)/logs/combined.log" ]; then \
		tail -f $(SERVER_DIR)/logs/combined.log; \
	else \
		echo "$(YELLOW)No log file found. Start the server first.$(NC)"; \
	fi

clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf $(SERVER_DIR)/dist
	rm -rf $(SERVER_DIR)/build
	rm -rf $(SERVER_DIR)/coverage
	rm -rf $(SERVER_DIR)/logs/*.log
	rm -rf $(CLIENT_DIR)/sender/build
	rm -rf $(CLIENT_DIR)/viewer/build
	find . -name "node_modules" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name ".DS_Store" -type f -delete 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

setup:
	@echo "$(GREEN)Setting up project...$(NC)"
	@echo "Creating directory structure..."
	@mkdir -p $(SERVER_DIR)/logs
	@mkdir -p $(CLIENT_DIR)/sender
	@mkdir -p $(CLIENT_DIR)/viewer
	@echo "Copying environment files..."
	@if [ -f "$(SERVER_DIR)/.env.example" ] && [ ! -f "$(SERVER_DIR)/.env" ]; then \
		cp $(SERVER_DIR)/.env.example $(SERVER_DIR)/.env; \
		echo "$(YELLOW)Created .env file - please update with your configuration$(NC)"; \
	fi
	@echo "$(GREEN)Setup complete!$(NC)"

# Health check
health:
	@echo "$(GREEN)Checking server health...$(NC)"
	@curl -s http://localhost:3000/health | jq '.' || echo "$(RED)Server is not responding$(NC)"

# Statistics
stats:
	@echo "$(GREEN)Server statistics:$(NC)"
	@curl -s http://localhost:3000/api/stats | jq '.' || echo "$(RED)Could not fetch statistics$(NC)"

# PM2 targets
pm2-start:
	cd $(SERVER_DIR) && pm2 start ecosystem.config.js

pm2-stop:
	cd $(SERVER_DIR) && pm2 stop ecosystem.config.js

pm2-restart:
	cd $(SERVER_DIR) && pm2 restart ecosystem.config.js

pm2-logs:
	pm2 logs video-streaming-server

# Security scan
security:
	@echo "$(GREEN)Running security audit...$(NC)"
	cd $(SERVER_DIR) && npm audit

security-fix:
	@echo "$(YELLOW)Attempting to fix security issues...$(NC)"
	cd $(SERVER_DIR) && npm audit fix