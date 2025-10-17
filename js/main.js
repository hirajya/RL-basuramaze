class Simulation {
    constructor() {
        this.env = new Environment('gameCanvas');
        // Make environment accessible to agents
        window.env = this.env;
        
        this.agent = null;
        this.chartUpdateTimeout = null;
        this.chart = this.initChart();
        this.running = false;
        this.episodeCount = 0;
        this.totalReward = 0;
        this.initEventListeners();
    }

    initChart() {
        const ctx = document.getElementById('rewardChart').getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Episode Reward',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    fill: false
                }, {
                    label: 'Average Reward (last 10)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                animation: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            maxTicksLimit: 10
                        }
                    },
                    x: {
                        ticks: {
                            maxTicksLimit: 20
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }

    updateChart(reward) {
        try {
            // Add new reward data
            this.chart.data.labels.push(this.episodeCount);
            this.chart.data.datasets[0].data.push(reward);

            // Calculate and update moving average
            const recentRewards = this.chart.data.datasets[0].data.slice(-10);
            const average = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
            this.chart.data.datasets[1].data.push(average);

            // Keep last 50 points for better visualization
            if (this.chart.data.labels.length > 50) {
                this.chart.data.labels.shift();
                this.chart.data.datasets[0].data.shift();
                this.chart.data.datasets[1].data.shift();
            }

            this.chart.update('none'); // Update without animation for better performance
        } catch (error) {
            console.error('Error updating chart:', error);
        }
    }

    initEventListeners() {
        document.getElementById('startButton').addEventListener('click', () => {
            if (!this.running) {
                console.log('Starting training...');
            }
            this.toggleSimulation();
        });
        
        document.getElementById('resetButton').addEventListener('click', () => {
            console.log('Resetting simulation...');
            this.reset();
        });
        
        document.getElementById('algorithm').addEventListener('change', (e) => {
            console.log('Switching to algorithm:', e.target.value);
            if (this.agent) {
                this.reset();
            }
        });
        
        ['learning_rate', 'gamma', 'epsilon', 'speed'].forEach(param => {
            const element = document.getElementById(param);
            const valueDisplay = document.getElementById(`${param}_value`);
            element.addEventListener('input', () => {
                valueDisplay.textContent = element.value;
                if (this.agent) {
                    console.log('Updating parameter:', param, 'to', element.value);
                    this.agent.updateParams(this.getParams());
                }
            });
        });

        // Add max steps handler
        document.getElementById('max_steps').addEventListener('input', (e) => {
            this.env.setMaxSteps(parseInt(e.target.value));
        });

        // Handle algorithm change to show/hide appropriate parameters
        document.getElementById('algorithm').addEventListener('change', (e) => {
            const isActorCritic = e.target.value === 'actor_critic';
            document.getElementById('default_params').style.display = isActorCritic ? 'none' : 'block';
            document.getElementById('actor_critic_params').style.display = isActorCritic ? 'block' : 'none';
            if (this.agent) {
                this.reset();
            }
        });

        // Add actor-critic parameter handlers
        ['actor_learning_rate', 'critic_learning_rate', 'ac_gamma', 'temperature'].forEach(param => {
            const element = document.getElementById(param);
            const valueDisplay = document.getElementById(`${param}_value`);
            element.addEventListener('input', () => {
                valueDisplay.textContent = element.value;
                if (this.agent) {
                    this.agent.updateParams(this.getParams());
                }
            });
        });
    }

    getParams() {
        const algorithm = document.getElementById('algorithm').value;
        
        if (algorithm === 'actor_critic') {
            return {
                actorLearningRate: parseFloat(document.getElementById('actor_learning_rate').value),
                criticLearningRate: parseFloat(document.getElementById('critic_learning_rate').value),
                gamma: parseFloat(document.getElementById('ac_gamma').value),
                temperature: parseFloat(document.getElementById('temperature').value)
            };
        }

        return {
            learningRate: parseFloat(document.getElementById('learning_rate').value),
            gamma: parseFloat(document.getElementById('gamma').value),
            epsilon: parseFloat(document.getElementById('epsilon').value)
        };
    }

    createAgent() {
        try {
            const algorithm = document.getElementById('algorithm').value;
            const params = this.getParams();
            console.log('Creating agent:', algorithm, params);

            let agent;
            switch(algorithm) {
                case 'monte_carlo':
                    agent = new MonteCarloAgent(params);
                    break;
                case 'q_learning':
                    agent = new QLearningAgent(params);
                    break;
                case 'actor_critic':
                    agent = new ActorCriticAgent(params);
                    break;
                default:
                    console.warn('Unknown algorithm:', algorithm, 'defaulting to Q-Learning');
                    agent = new QLearningAgent(params);
            }

            // Initialize agent
            agent.reset();
            
            // Update environment with new algorithm
            this.env.setAlgorithm(algorithm);
            
            console.log('Agent created successfully:', algorithm);
            return agent;
        } catch (error) {
            console.error('Error creating agent:', error);
            // Default to Q-Learning if agent creation fails
            return new QLearningAgent(this.getParams());
        }
    }

    toggleSimulation() {
        try {
            if (this.running) {
                console.log('Pausing simulation');
                this.running = false;
                document.getElementById('startButton').textContent = 'Start Training';
                this.env.enableEditing();
            } else {
                console.log('Starting simulation');
                
                // Validate grid setup first
                if (!this.env.checkInitialization()) {
                    console.log('Grid initialization failed, remaining in edit mode');
                    return;
                }
                
                // Create fresh agent and start training
                this.agent = this.createAgent();
                if (!this.agent) {
                    alert('Failed to initialize agent. Please try a different algorithm.');
                    return;
                }

                // Initialize status panel
                document.getElementById('episodeCounter').textContent = '0';
                document.getElementById('totalEpisodes').textContent = document.getElementById('episodes').value;
                document.getElementById('currentReward').textContent = '0';
                document.getElementById('bestReward').textContent = '0';
                document.getElementById('averageReward').textContent = '0';
                document.getElementById('trashCount').textContent = this.env.trashCount;
                document.getElementById('successfulCompletions').textContent = this.env.successfulCompletions;

                this.running = true;
                document.getElementById('startButton').textContent = 'Pause Training';
                this.env.disableEditing();
                requestAnimationFrame(() => this.runEpisode());
            }
        } catch (error) {
            console.error('Error in toggleSimulation:', error);
            this.running = false;
            document.getElementById('startButton').textContent = 'Start Training';
            this.env.enableEditing();
            alert('An error occurred while starting the training. Please check the console for details.');
        }
    }

    reset() {
        this.running = false;
        document.getElementById('startButton').textContent = 'Start Training';
        this.agent = this.createAgent();
        this.episodeCount = 0;
        this.totalReward = 0;
        
        // Reset successful completions counter
        this.env.successfulCompletions = 0;
        
        this.env.resetHeatmap();
        this.env.reset();
        this.env.enableEditing(); // Enable editing after reset
        this.chart.data.labels = [];
        this.chart.data.datasets[0].data = [];
        this.chart.data.datasets[1].data = [];
        this.chart.update();
        
        // Update status panel to reflect reset
        this.updateStatusPanel();
    }

    updateStatusPanel() {
        try {
            document.getElementById('episodeCounter').textContent = this.episodeCount;
            document.getElementById('totalEpisodes').textContent = document.getElementById('episodes').value;
            document.getElementById('currentReward').textContent = this.totalReward.toFixed(1);
            document.getElementById('trashCount').textContent = this.env.trashCount;
            
            // Update successful completions counter
            document.getElementById('successfulCompletions').textContent = this.env.successfulCompletions;

            // Update best and average rewards
            const allRewards = this.chart.data.datasets[0].data;
            if (allRewards.length > 0) {
                const bestReward = Math.max(...allRewards);
                const recentRewards = allRewards.slice(-10);
                const avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
                
                document.getElementById('bestReward').textContent = bestReward.toFixed(1);
                document.getElementById('averageReward').textContent = avgReward.toFixed(1);
            }

            // Trigger a render to update heatmap
            this.env.render();
        } catch (error) {
            console.error('Error updating status panel:', error);
        }
    }

    async runEpisode() {
        try {
            if (!this.running) return;
            if (!this.agent) {
                console.error('No agent initialized!');
                this.running = false;
                document.getElementById('startButton').textContent = 'Start Training';
                return;
            }

            this.episodeCount++;
            let state = this.env.reset();
            this.totalReward = 0;
            let done = false;

            // Update initial status
            this.updateStatusPanel();

            while (!done && this.running) {
                try {
                    // Get and execute action
                    const action = this.agent.selectAction(state);
                    const [nextState, reward, isDone] = this.env.step(action);
                    
                    // Update agent
                    const algorithm = document.getElementById('algorithm').value;
                    if (algorithm === 'monte_carlo') {
                        this.agent.update(state, action, reward);
                    } else if (algorithm === 'q_learning') {
                        this.agent.update(state, action, reward, nextState);
                    } else {  // actor_critic
                        this.agent.update(state, action, reward, nextState, isDone);
                    }

                    this.totalReward += reward;
                    state = nextState;
                    done = isDone;

                    // Update status and visualization
                    this.updateStatusPanel();

                    // Control speed
                    const speed = parseInt(document.getElementById('speed').value);
                    if (speed < 60) {
                        await new Promise(r => setTimeout(r, 1000 / speed));
                    } else {
                        await new Promise(r => requestAnimationFrame(r));
                    }
                } catch (stepError) {
                    console.error('Error during episode step:', stepError);
                    done = true;
                }
            }

            // Episode cleanup
            if (document.getElementById('algorithm').value === 'monte_carlo') {
                this.agent.episodeEnd();
            }

            // Update chart with a slight delay to prevent performance issues
            if (this.chartUpdateTimeout) {
                clearTimeout(this.chartUpdateTimeout);
            }
            this.chartUpdateTimeout = setTimeout(() => {
                this.updateChart(this.totalReward);
            }, 16);

            // Continue to next episode or stop
            if (this.running && this.episodeCount < parseInt(document.getElementById('episodes').value)) {
                requestAnimationFrame(() => this.runEpisode());
            } else {
                this.running = false;
                document.getElementById('startButton').textContent = 'Start Training';
                this.env.enableEditing();
            }
        } catch (error) {
            console.error('Fatal error in runEpisode:', error);
            this.running = false;
            document.getElementById('startButton').textContent = 'Start Training';
            this.env.enableEditing();
        }
    }
}

// Start simulation when page loads
window.addEventListener('load', () => {
    const simulation = new Simulation();
});