// server/src/services/connectionManager.js - Manages WebSocket connections
const logger = require('../utils/logger');
const EventEmitter = require('events');

class ConnectionManager extends EventEmitter {
    constructor() {
        super();
        this.connections = new Map();
        this.connectionsByType = {
            sender: new Set(),
            viewer: new Set()
        };
    }
    
    generateClientId() {
        return Math.random().toString(36).substring(2, 15) + 
               Date.now().toString(36);
    }
    
    addConnection(clientId, connectionData) {
        this.connections.set(clientId, connectionData);
        
        if (connectionData.type) {
            this.connectionsByType[connectionData.type].add(clientId);
        }
        
        this.emit('connection:added', { clientId, ...connectionData });
        logger.debug(`Connection added: ${clientId}`);
        
        return connectionData;
    }
    
    updateConnection(clientId, updates) {
        const connection = this.connections.get(clientId);
        if (!connection) {
            logger.warn(`Attempted to update non-existent connection: ${clientId}`);
            return null;
        }
        
        // Handle type changes
        if (updates.type && updates.type !== connection.type) {
            // Remove from old type set
            if (connection.type) {
                this.connectionsByType[connection.type].delete(clientId);
            }
            // Add to new type set
            this.connectionsByType[updates.type].add(clientId);
        }
        
        // Update connection data
        Object.assign(connection, updates);
        
        this.emit('connection:updated', { clientId, updates });
        logger.debug(`Connection updated: ${clientId}`);
        
        return connection;
    }
    
    removeConnection(clientId) {
        const connection = this.connections.get(clientId);
        if (!connection) return null;
        
        // Remove from type set
        if (connection.type) {
            this.connectionsByType[connection.type].delete(clientId);
        }
        
        this.connections.delete(clientId);
        
        this.emit('connection:removed', { clientId, connection });
        logger.debug(`Connection removed: ${clientId}`);
        
        // Clean up any associated feeds (handled by feedManager via events)
        if (connection.type === 'sender' && connection.feedId) {
            const feedManager = require('./feedManager');
            feedManager.handleSenderDisconnect(clientId, connection.feedId);
        }
        
        return connection;
    }
    
    getConnection(clientId) {
        return this.connections.get(clientId);
    }
    
    getConnectionsByType(type) {
        const ids = this.connectionsByType[type];
        if (!ids) return [];
        
        return Array.from(ids).map(id => this.connections.get(id)).filter(Boolean);
    }
    
    getAllConnections() {
        return Array.from(this.connections.values());
    }
    
    getConnectionStats() {
        const now = Date.now();
        const connections = this.getAllConnections();
        
        return {
            total: connections.length,
            byType: {
                sender: this.connectionsByType.sender.size,
                viewer: this.connectionsByType.viewer.size,
                unregistered: connections.filter(c => !c.type).length
            },
            averageConnectionTime: connections.length > 0 
                ? connections.reduce((sum, c) => {
                    const joinTime = new Date(c.timestamp).getTime();
                    return sum + (now - joinTime);
                }, 0) / connections.length / 1000 // Convert to seconds
                : 0,
            byIP: this.getConnectionsByIP()
        };
    }
    
    getConnectionsByIP() {
        const byIP = {};
        this.connections.forEach(connection => {
            const ip = connection.ip;
            if (!byIP[ip]) {
                byIP[ip] = 0;
            }
            byIP[ip]++;
        });
        return byIP;
    }
    
    isConnected(clientId) {
        const connection = this.connections.get(clientId);
        return connection && connection.ws && connection.ws.readyState === 1; // WebSocket.OPEN
    }
    
    broadcastToType(type, message) {
        const connections = this.getConnectionsByType(type);
        const wsServer = require('../websocket/wsServer');
        
        connections.forEach(connection => {
            if (this.isConnected(connection.id)) {
                wsServer.sendToClient(connection.id, message);
            }
        });
    }
    
    cleanup() {
        // Remove disconnected clients
        const disconnected = [];
        this.connections.forEach((connection, clientId) => {
            if (!this.isConnected(clientId)) {
                disconnected.push(clientId);
            }
        });
        
        disconnected.forEach(clientId => {
            logger.info(`Cleaning up disconnected client: ${clientId}`);
            this.removeConnection(clientId);
        });
        
        return disconnected.length;
    }
}

// Singleton instance
const connectionManager = new ConnectionManager();

// Periodic cleanup
setInterval(() => {
    const cleaned = connectionManager.cleanup();
    if (cleaned > 0) {
        logger.info(`Cleaned up ${cleaned} disconnected connections`);
    }
}, 30000); // Every 30 seconds

module.exports = connectionManager;