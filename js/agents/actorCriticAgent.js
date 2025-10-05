class ActorCriticAgent extends BaseAgent {
    constructor(params) {
        super(params);
        
        // Store separate learning rates for actor and critic
        this.actorLearningRate = params.actorLearningRate || 0.001;
        this.criticLearningRate = params.criticLearningRate || 0.01;
        this.temperature = params.temperature || 1.0;
        
        // Initialize networks
        this.actorNetwork = new NeuralNetwork([24, 64, 32, 4]);  // Deeper network
        this.criticNetwork = new NeuralNetwork([24, 64, 32, 1]);
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.episodeCount = 0;

        // Add state value tracking
        this.minStateValue = -20;
        this.maxStateValue = 50;
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
            const actionProbs = this.actorNetwork.forward(input);
            
            // Get and normalize state value
            const stateValue = this.criticNetwork.forward(input)[0];
            const normalizedValue = Math.max(this.minStateValue, Math.min(this.maxStateValue, stateValue));
            
            // Update state value in environment
            window.env.updateStateValue(state.wallE.x, state.wallE.y, normalizedValue);
            
            // Apply softmax with temperature
            const expProbs = actionProbs.map(x => Math.exp(x / this.temperature));
            const sum = expProbs.reduce((a, b) => a + b, 0);
            const probabilities = expProbs.map(x => x / sum);

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
            return probabilities.length - 1;  // Fallback to last action
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

    episodeEnd() {
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