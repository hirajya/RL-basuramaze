class ActorCriticAgent extends BaseAgent {
    constructor(params) {
        super(params);
        
        // Store separate learning rates for actor and critic
        this.actorLearningRate = params.actorLearningRate || 0.01;
        this.criticLearningRate = params.criticLearningRate || 0.01;
        this.temperature = params.temperature || 1.0;
        
        // Initialize networks
        this.actorNetwork = new NeuralNetwork([24, 64, 32, 4]);
        this.criticNetwork = new NeuralNetwork([24, 64, 32, 1]);
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.episodeCount = 0;

        // Add state value tracking
        this.minStateValue = -20;
        this.maxStateValue = 50;
        
        // Track episode transitions for table export with episode reset column
        this.episodeTransitions = [];
        this.currentEpisode = 0;
        this.updateHistory = []; // Track all updates with episode reset column
    }

    stateToTensor(state) {
        try {
            if (!state || !state.wallE || !state.evilRobot || typeof state.trashCount !== 'number') {
                console.error('Invalid state object:', state);
                return new Array(24).fill(0);
            }
            
            const input = new Array(24).fill(0);
            
            // Wall-E position (one-hot)
            const wallEIndex = state.wallE.y * 6 + state.wallE.x;
            input[wallEIndex] = 1;
            
            // Evil robot position (one-hot)
            const robotIndex = 12 + state.evilRobot.y * 6 + state.evilRobot.x;
            input[robotIndex] = 1;
            
            // Add trash count information
            input[23] = state.trashCount / 3;  // Normalize by max possible trash
            
            return input;
        } catch (error) {
            console.error('Error in stateToTensor:', error);
            return new Array(24).fill(0);
        }
    }

    selectAction(state) {
        try {
            const input = this.stateToTensor(state);
            const actionLogits = this.actorNetwork.forward(input);
            
            // Check for NaN values in network output
            if (actionLogits.some(val => isNaN(val))) {
                console.warn('NaN detected in actor network output, using random action');
                return Math.floor(Math.random() * this.actionSpace);
            }
            
            // Get and normalize state value
            const stateValue = this.criticNetwork.forward(input)[0];
            if (!isNaN(stateValue)) {
                const normalizedValue = Math.max(this.minStateValue, Math.min(this.maxStateValue, stateValue));
                window.env.updateStateValue(state.wallE.x, state.wallE.y, normalizedValue);
            }
            
            // Apply softmax with temperature - fix NaN issues
            const maxLogit = Math.max(...actionLogits);
            const expLogits = actionLogits.map(x => Math.exp((x - maxLogit) / this.temperature));
            const sum = expLogits.reduce((a, b) => a + b, 0);
            
            // Prevent division by zero
            if (sum === 0 || isNaN(sum)) {
                console.warn('Invalid softmax sum, using random action');
                return Math.floor(Math.random() * this.actionSpace);
            }
            
            const probabilities = expLogits.map(x => x / sum);
            
            // Check for NaN probabilities
            if (probabilities.some(p => isNaN(p))) {
                console.warn('NaN probabilities detected, using random action');
                return Math.floor(Math.random() * this.actionSpace);
            }

            // Epsilon-greedy exploration
            if (Math.random() < this.epsilon) {
                return Math.floor(Math.random() * this.actionSpace);
            }

            // Sample from policy using roulette wheel selection
            const rand = Math.random();
            let cumSum = 0;
            for (let i = 0; i < probabilities.length; i++) {
                cumSum += probabilities[i];
                if (rand < cumSum) {
                    return i;
                }
            }
            return probabilities.length - 1;
        } catch (error) {
            console.error('Error in selectAction:', error);
            return Math.floor(Math.random() * this.actionSpace);
        }
    }

    update(state, action, reward, nextState, done) {
        if (!state || !nextState || !Number.isInteger(action) || typeof reward !== 'number') {
            console.error('Invalid input to update:', { state, action, reward, nextState });
            return;
        }

        try {
            // Store transition
            this.states.push(this.stateToTensor(state));
            this.actions.push(action);
            this.rewards.push(reward);

            if (done) {
                this.episodeEnd();
            }
        } catch (error) {
            console.error('Error in update:', error);
        }
    }

    // Set current episode number (called from simulation)
    setEpisode(episodeNum) {
        this.currentEpisode = episodeNum;
        // Add simple episode reset marker to update history
        this.updateHistory.push({
            episode: episodeNum,
            step: 'RESET',
            state: `Episode_${episodeNum}_Start`,
            action: 'RESET',
            stateValue: '0.0000',
            reward: 'WALL-E_RESET',
            timestamp: new Date().toISOString(),
            episodeReset: 'TRUE'  // Clear indicator for episode reset
        });
    }

    episodeEnd() {
        // Add episode end marker before processing
        if (this.currentEpisode > 0) {
            this.episodeTransitions.push({
                episode: this.currentEpisode,
                marker: `EPISODE_${this.currentEpisode}_END`,
                timestamp: new Date().toISOString(),
                type: 'END'
            });
        }

        try {
            // Calculate returns and advantages
            const values = this.states.map(state => this.criticNetwork.forward(state)[0]);
            const returns = [];
            let G = 0;
            
            // Calculate returns in reverse
            for (let t = this.rewards.length - 1; t >= 0; t--) {
                G = this.rewards[t] + this.gamma * G;
                returns.unshift(G);
                
                // Track min/max state values
                this.minStateValue = Math.min(this.minStateValue, G);
                this.maxStateValue = Math.max(this.maxStateValue, G);
            }

            // Normalize returns for stable training
            const minReturn = Math.min(...returns);
            const maxReturn = Math.max(...returns);
            const normalizedReturns = returns.map(G => 
                (G - minReturn) / (maxReturn - minReturn + 1e-8) * 70 - 20
            );

            // Calculate advantages
            const advantages = normalizedReturns.map((G, i) => G - values[i]);

            // Clip advantages for stability
            const clippedAdvantages = advantages.map(a => 
                Math.max(Math.min(a, 10), -10)
            );

            // Track value updates for export with Episode_Reset column
            for (let t = 0; t < this.states.length; t++) {
                const stateStr = this.stateToString({
                    wallE: { x: 0, y: 0 }, // This would need actual state reconstruction
                    evilRobot: { x: 0, y: 0 },
                    trashCount: 0
                });
                
                this.updateHistory.push({
                    episode: this.currentEpisode,
                    step: t + 1,
                    state: `Step_${t + 1}`,
                    action: this.actions[t],
                    stateValue: normalizedReturns[t].toFixed(4),
                    reward: this.rewards[t],
                    timestamp: new Date().toISOString(),
                    episodeReset: 'FALSE'  // Normal update, not a reset
                });
            }

            // Update networks with mini-batch
            const batchSize = Math.min(32, this.states.length);
            for (let i = 0; i < this.states.length; i += batchSize) {
                const batchEnd = Math.min(i + batchSize, this.states.length);
                this.updateNetworks(
                    this.states.slice(i, batchEnd),
                    this.actions.slice(i, batchEnd),
                    normalizedReturns.slice(i, batchEnd),
                    clippedAdvantages.slice(i, batchEnd)
                );
            }

            // Log episode statistics
            this.episodeCount++;
            const totalReward = this.rewards.reduce((a, b) => a + b, 0);
            this.addReward(totalReward);

            // Clear episode data
            this.states = [];
            this.actions = [];
            this.rewards = [];
        } catch (error) {
            console.error('Error in episodeEnd:', error);
            // Reset episode data on error
            this.states = [];
            this.actions = [];
            this.rewards = [];
        }
    }

    updateNetworks(states, actions, returns, advantages) {
        // Update critic
        for (let i = 0; i < states.length; i++) {
            this.criticNetwork.backward(
                states[i],
                [returns[i]],
                this.criticLearningRate
            );
        }

        // Update actor
        for (let i = 0; i < states.length; i++) {
            const actionProbs = this.actorNetwork.forward(states[i]);
            const actionGradients = actionProbs.map((_, j) => 
                j === actions[i] ? advantages[i] / this.temperature : 0
            );
            this.actorNetwork.backward(
                states[i],
                actionGradients,
                this.actorLearningRate
            );
        }
    }

    updateParams(params) {
        super.updateParams(params);
        if (params.actorLearningRate !== undefined) {
            this.actorLearningRate = params.actorLearningRate;
        }
        if (params.criticLearningRate !== undefined) {
            this.criticLearningRate = params.criticLearningRate;
        }
        if (params.temperature !== undefined) {
            this.temperature = params.temperature;
        }
    }

    reset() {
        super.reset();
        this.actorNetwork = new NeuralNetwork([24, 64, 32, 4]);
        this.criticNetwork = new NeuralNetwork([24, 64, 32, 1]);
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.episodeCount = 0;
        this.minStateValue = -20;
        this.maxStateValue = 50;
        this.episodeTransitions = [];
        this.currentEpisode = 0;
        this.updateHistory = []; // Reset update history
    }

    // Export value table with Episode_Reset column
    exportValueTable() {
        // Export update history with Episode_Reset column
        const csvData = ['Episode,Step,State,Action,State_Value,Reward,Episode_Reset,Timestamp'];
        
        this.updateHistory.forEach(entry => {
            csvData.push(`${entry.episode},${entry.step},${entry.state},${entry.action},${entry.stateValue},${entry.reward || ''},${entry.episodeReset || 'FALSE'},${entry.timestamp}`);
        });
        
        const csvContent = csvData.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `actor_critic_values_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        
        URL.revokeObjectURL(url);

        // ...rest of existing valueData logic...
        const valueData = [];
        
        // Add episode transition markers at the beginning
        for (const transition of this.episodeTransitions) {
            valueData.push({
                state: transition.marker,
                wallE_x: `--- ${transition.type} ---`,
                wallE_y: `Episode ${transition.episode}`,
                evil_x: '---',
                evil_y: '---',
                trashCount: '---',
                stateValue: '0.0000',
                episodeMarker: transition.marker,
                timestamp: transition.timestamp
            });
        }
        
        // Only export states that have been actually encountered during training
        // We'll sample states that produce non-uniform probabilities
        const meaningfulStates = [];
        
        for (let wallE_x = 0; wallE_x < 6; wallE_x++) {
            for (let wallE_y = 0; wallE_y < 6; wallE_y++) {
                for (let evil_x = 0; evil_x < 6; evil_x++) {
                    for (let evil_y = 0; evil_y < 6; evil_y++) {
                        for (let trashCount = 0; trashCount <= 3; trashCount++) {
                            const state = {
                                wallE: { x: wallE_x, y: wallE_y },
                                evilRobot: { x: evil_x, y: evil_y },
                                trashCount: trashCount
                            };
                            
                            const input = this.stateToTensor(state);
                            const actionLogits = this.actorNetwork.forward(input);
                            
                            // Check if this state has learned meaningful probabilities
                            if (!actionLogits.some(val => isNaN(val))) {
                                const maxLogit = Math.max(...actionLogits);
                                const expLogits = actionLogits.map(x => Math.exp((x - maxLogit) / this.temperature));
                                const sum = expLogits.reduce((a, b) => a + b, 0);
                                
                                if (sum > 0 && !isNaN(sum)) {
                                    const probabilities = expLogits.map(x => x / sum);
                                    
                                    // Check if probabilities are meaningfully different from uniform (0.25 each)
                                    const maxProb = Math.max(...probabilities);
                                    const minProb = Math.min(...probabilities);
                                    
                                    // Only include if there's significant learning (not uniform distribution)
                                    if (maxProb - minProb > 0.1) { // Threshold for meaningful learning
                                        const stateStr = this.stateToString(state);
                                        let stateValue = this.criticNetwork.forward(input)[0];
                                        
                                        if (isNaN(stateValue)) {
                                            stateValue = 0;
                                        }
                                        
                                        valueData.push({
                                            state: stateStr,
                                            wallE_x: wallE_x,
                                            wallE_y: wallE_y,
                                            evil_x: evil_x,
                                            evil_y: evil_y,
                                            trashCount: trashCount,
                                            stateValue: stateValue.toFixed(4),
                                            episodeMarker: '',
                                            timestamp: ''
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return valueData;
    }

    // Export policy table from actor network with episode transition markers
    exportPolicyTable() {
        const policyData = [];
        const actionNames = ['Up', 'Right', 'Down', 'Left'];
        
        // Add episode transition markers at the beginning
        for (const transition of this.episodeTransitions) {
            policyData.push({
                state: transition.marker,
                wallE_x: `--- ${transition.type} ---`,
                wallE_y: `Episode ${transition.episode}`,
                evil_x: '---',
                evil_y: '---',
                trashCount: '---',
                optimalAction: '---',
                optimalActionName: transition.marker,
                actionProbUp: '0.0000',
                actionProbRight: '0.0000',
                actionProbDown: '0.0000',
                actionProbLeft: '0.0000',
                maxProbability: '0.0000',
                episodeMarker: transition.marker,
                timestamp: transition.timestamp
            });
        }
        
        for (let wallE_x = 0; wallE_x < 6; wallE_x++) {
            for (let wallE_y = 0; wallE_y < 6; wallE_y++) {
                for (let evil_x = 0; evil_x < 6; evil_x++) {
                    for (let evil_y = 0; evil_y < 6; evil_y++) {
                        for (let trashCount = 0; trashCount <= 3; trashCount++) {
                            const state = {
                                wallE: { x: wallE_x, y: wallE_y },
                                evilRobot: { x: evil_x, y: evil_y },
                                trashCount: trashCount
                            };
                            
                            const stateStr = this.stateToString(state);
                            const input = this.stateToTensor(state);
                            const actionLogits = this.actorNetwork.forward(input);
                            
                            let probabilities = [0.25, 0.25, 0.25, 0.25]; // Default uniform distribution
                            
                            try {
                                if (!actionLogits.some(val => isNaN(val))) {
                                    const maxLogit = Math.max(...actionLogits);
                                    const expLogits = actionLogits.map(x => Math.exp((x - maxLogit) / this.temperature));
                                    const sum = expLogits.reduce((a, b) => a + b, 0);
                                    
                                    if (sum > 0 && !isNaN(sum)) {
                                        probabilities = expLogits.map(x => x / sum);
                                        
                                        // Final check for NaN
                                        if (probabilities.some(p => isNaN(p))) {
                                            probabilities = [0.25, 0.25, 0.25, 0.25];
                                        }
                                    }
                                }
                            } catch (error) {
                                probabilities = [0.25, 0.25, 0.25, 0.25];
                            }
                            
                            // Only export states with meaningful learning (not uniform distribution)
                            const maxProb = Math.max(...probabilities);
                            const minProb = Math.min(...probabilities);
                            
                            if (maxProb - minProb > 0.1) { // Threshold for meaningful learning
                                const maxProbIndex = probabilities.indexOf(Math.max(...probabilities));
                                
                                policyData.push({
                                    state: stateStr,
                                    wallE_x: wallE_x,
                                    wallE_y: wallE_y,
                                    evil_x: evil_x,
                                    evil_y: evil_y,
                                    trashCount: trashCount,
                                    optimalAction: maxProbIndex,
                                    optimalActionName: actionNames[maxProbIndex],
                                    actionProbUp: probabilities[0].toFixed(4),
                                    actionProbRight: probabilities[1].toFixed(4),
                                    actionProbDown: probabilities[2].toFixed(4),
                                    actionProbLeft: probabilities[3].toFixed(4),
                                    maxProbability: Math.max(...probabilities).toFixed(4),
                                    episodeMarker: '',
                                    timestamp: ''
                                });
                            }
                        }
                    }
                }
            }
        }
        
        return policyData;
    }

    // Get Actor-Critic table statistics
    getActorCriticStats() {
        const totalStates = 6 * 6 * 6 * 6 * 4; // All possible state combinations
        
        // Sample some states to get value range
        const sampleValues = [];
        for (let i = 0; i < 100; i++) {
            const state = {
                wallE: { x: Math.floor(Math.random() * 6), y: Math.floor(Math.random() * 6) },
                evilRobot: { x: Math.floor(Math.random() * 6), y: Math.floor(Math.random() * 6) },
                trashCount: Math.floor(Math.random() * 4)
            };
            const input = this.stateToTensor(state);
            const value = this.criticNetwork.forward(input)[0];
            if (!isNaN(value)) {
                sampleValues.push(value);
            }
        }
        
        return {
            totalStates: totalStates,
            networkArchitecture: 'Actor: [24,64,32,4], Critic: [24,64,32,1]',
            temperature: this.temperature.toFixed(4),
            minSampleValue: sampleValues.length > 0 ? Math.min(...sampleValues).toFixed(4) : '0',
            maxSampleValue: sampleValues.length > 0 ? Math.max(...sampleValues).toFixed(4) : '0',
            avgSampleValue: sampleValues.length > 0 ? (sampleValues.reduce((a, b) => a + b, 0) / sampleValues.length).toFixed(4) : '0'
        };
    }
}

// Simple neural network implementation
class NeuralNetwork {
    constructor(layers) {
        this.weights = [];
        this.biases = [];
        
        // Initialize weights and biases
        for (let i = 0; i < layers.length - 1; i++) {
            this.weights.push(
                Array(layers[i]).fill().map(() => 
                    Array(layers[i + 1]).fill().map(() => 
                        Math.random() * 2 - 1
                    )
                )
            );
            this.biases.push(Array(layers[i + 1]).fill(0));
        }
    }

    forward(input) {
        let current = input;
        
        // Forward propagation through layers
        for (let i = 0; i < this.weights.length; i++) {
            const layer = [];
            for (let j = 0; j < this.weights[i][0].length; j++) {
                let sum = this.biases[i][j];
                for (let k = 0; k < this.weights[i].length; k++) {
                    sum += current[k] * this.weights[i][k][j];
                }
                layer.push(i === this.weights.length - 1 ? sum : Math.max(0, sum)); // ReLU except last layer
            }
            current = layer;
        }
        
        return current;
    }

    backward(input, gradients, learningRate) {
        // Simple SGD update
        const output = this.forward(input);
        
        for (let i = this.weights.length - 1; i >= 0; i--) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    const grad = gradients[k] * input[j];
                    this.weights[i][j][k] += learningRate * grad;
                }
            }
            
            for (let j = 0; j < this.biases[i].length; j++) {
                this.biases[i][j] += learningRate * gradients[j];
            }
        }
    }
}