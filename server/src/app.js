// server/src/app.js - Express application setup
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const path = require('path');

const config = require('./config');
const logger = require('./utils/logger');
const errorHandler = require('./middleware/errorHandler');
const rateLimiter = require('./middleware/rateLimiter');

// Import routes
const healthRoutes = require('./routes/health');
const feedRoutes = require('./routes/feeds');
const statsRoutes = require('./routes/stats');

const app = express();

// Security middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            connectSrc: ["'self'", "wss:", "ws:"],
        },
    },
}));

// CORS configuration
app.use(cors({
    origin: config.cors.origins,
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
}));

// Compression middleware
app.use(compression());

// Request logging
if (config.nodeEnv !== 'test') {
    app.use(morgan('combined', {
        stream: { write: message => logger.info(message.trim()) }
    }));
}

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting
app.use('/api/', rateLimiter);

// Static files (if needed for viewer client)
if (config.serveStatic) {
    app.use(express.static(path.join(__dirname, '../../client/viewer/build')));
}

// API Routes
app.use('/health', healthRoutes);
app.use('/api/feeds', feedRoutes);
app.use('/api/stats', statsRoutes);

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        name: 'Video Streaming Server',
        version: config.version,
        status: 'operational',
        endpoints: {
            health: '/health',
            feeds: '/api/feeds',
            stats: '/api/stats',
            websocket: `${config.isProduction ? 'wss' : 'ws'}://${req.get('host')}`
        }
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        error: 'Not Found',
        message: `Cannot ${req.method} ${req.url}`,
        status: 404
    });
});

// Error handling middleware (must be last)
app.use(errorHandler);

module.exports = app;