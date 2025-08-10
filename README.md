# Video Streaming Platform

A robust, scalable WebRTC-based video streaming platform designed for hosting on AWS. Supports multiple simultaneous video feeds with low-latency streaming.

## Features

- **Multi-Feed Support**: Handle multiple video streams simultaneously
- **WebRTC Technology**: Low-latency peer-to-peer video streaming
- **Scalable Architecture**: Horizontal scaling with Redis support
- **Real-time Signaling**: WebSocket-based signaling server
- **AWS Ready**: Optimized for deployment on AWS infrastructure
- **Monitoring**: Built-in health checks and metrics endpoints
- **Security**: Rate limiting, CORS, helmet protection
- **Docker Support**: Containerized deployment ready

## Prerequisites

- Node.js 18+ 
- npm 9+
- Docker & Docker Compose (optional)
- Redis (optional, for scaling)
- AWS Account (for production deployment)

## Installation

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/video-streaming-platform.git
cd video-streaming-platform
```

2. **Install dependencies**
```bash
cd server
npm install
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Start the server**
```bash
# Development mode with hot reload
npm run dev

# Production mode
npm start
```

### Docker Deployment

1. **Build and run with Docker Compose**
```bash
cd infrastructure/docker
docker-compose up -d
```

2. **View logs**
```bash
docker-compose logs -f video-server
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Sender    │────▶│   Server    │────▶│   Viewer    │
│   Client    │     │  (AWS EC2)  │     │   Client    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                    │
       │              WebSocket                 │
       │               Signaling                │
       │                   │                    │
       └───────── WebRTC Connection ───────────┘
                    (Peer-to-Peer)
```

## API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check |
| GET | `/health/detailed` | Detailed health status |
| GET | `/api/feeds` | List all active feeds |
| GET | `/api/feeds/:id` | Get specific feed details |
| GET | `/api/feeds/:id/stats` | Get feed statistics |
| DELETE | `/api/feeds/:id` | Stop a feed |
| GET | `/api/stats` | Server statistics |

### WebSocket Events

**Client → Server:**
- `register-sender` - Register as video sender
- `register-viewer` - Register as viewer
- `request-feed` - Request to watch a feed
- `offer` - WebRTC offer
- `answer` - WebRTC answer
- `ice-candidate` - ICE candidate

**Server → Client:**
- `connection` - Connection established
- `registered` - Registration confirmed
- `feed-ready` - Feed ready to stream
- `feeds-update` - Feed list updated

## AWS Deployment

### Using EC2

1. **Launch EC2 Instance**
   - Use Amazon Linux 2 or Ubuntu 20.04+
   - Instance type: t3.medium or larger
   - Security group: Open ports 3000 (app), 80/443 (web)

2. **Install Dependencies**
```bash
# Install Node.js
curl -sL https://rpm.nodesource.com/setup_18.x | sudo bash -
sudo yum install nodejs -y

# Install PM2
sudo npm install -g pm2
```

3. **Deploy Application**
```bash
# Clone and setup
git clone <repository>
cd video-streaming-platform/server
npm install --production

# Start with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

### Using ECS/Fargate

1. **Build Docker Image**
```bash
docker build -t video-streaming-server .
```

2. **Push to ECR**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ecr-uri>
docker tag video-streaming-server:latest <ecr-uri>/video-streaming-server:latest
docker push <ecr-uri>/video-streaming-server:latest
```

3. **Create ECS Service**
   - Use the provided Terraform configurations in `/infrastructure/terraform`

### Load Balancing

Configure Application Load Balancer (ALB):
- Enable WebSocket support
- Health check: `/health`
- Stickiness: Enable for WebSocket connections

## 🔧 Configuration

### Environment Variables

Key configuration options:

```bash
# Server
PORT=3000
NODE_ENV=production

# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# WebRTC
TURN_SERVER=turn:your-turn.com:3478
TURN_USERNAME=username
TURN_PASSWORD=password

# Redis (for scaling)
REDIS_ENABLED=true
REDIS_URL=redis://your-redis-cluster.aws.com:6379
```

### Scaling Configuration

For horizontal scaling:
1. Enable Redis for session management
2. Deploy multiple server instances
3. Use ALB with sticky sessions
4. Consider using ElastiCache for Redis

## 📊 Monitoring

### CloudWatch Integration

Metrics are automatically sent to CloudWatch when enabled:
```bash
CLOUDWATCH_ENABLED=true
CLOUDWATCH_NAMESPACE=VideoStreaming
```

### Grafana Dashboards

Import the provided dashboards from `/monitoring/grafana/dashboards`

### Health Checks

- `/health` - Basic health
- `/health/live` - Kubernetes liveness
- `/health/ready` - Kubernetes readiness

## Testing

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Watch mode
npm run test:watch
```

## Security

- **Authentication**: JWT-based auth (when enabled)
- **Rate Limiting**: Configurable per-endpoint limits
- **CORS**: Configurable origins
- **Helmet**: Security headers
- **SSL/TLS**: Required for production WebRTC

## Documentation

- [API Documentation](docs/API.md)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Contributing](docs/CONTRIBUTING.md)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- WebRTC for real-time communication
- Socket.io/ws for WebSocket implementation
- Express.js for the server framework
- AWS for cloud infrastructure

## Support

For issues and questions:
- Open an issue on GitHub
- Email: support@example.com
- Documentation: [https://docs.example.com](https://docs.example.com)