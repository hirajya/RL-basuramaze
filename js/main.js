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
        
        // Comprehensive metrics tracking for CSV export
        this.episodeMetrics = [];
        this.currentEpisodeSteps = 0;
        this.episodeStartTime = null;
        
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
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    tension: 0.1,
                    fill: false,
                    pointRadius: 2,
                    pointHoverRadius: 4
                }, {
                    label: 'Average Reward (last 10)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.1,
                    fill: false,
                    pointRadius: 1,
                    pointHoverRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            maxTicksLimit: 8
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    },
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        ticks: {
                            maxTicksLimit: 10,
                            callback: function(value) {
                                return Math.floor(value);
                            }
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: true,
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x',
                        },
                        pan: {
                            enabled: true,
                            mode: 'x',
                        }
                    }
                }
            }
        });
    }

    updateChart(reward) {
        try {
            // Add new reward data
            this.chart.data.labels.push(this.episodeCount);
            this.chart.data.datasets[0].data.push({x: this.episodeCount, y: reward});

            // Calculate and update moving average
            const recentRewards = this.chart.data.datasets[0].data.slice(-10);
            const avgReward = recentRewards.reduce((a, b) => a + (b.y || b), 0) / recentRewards.length;
            this.chart.data.datasets[1].data.push({x: this.episodeCount, y: avgReward});

            // Dynamic compression based on episode count
            const maxVisiblePoints = 100; // Maximum points to show at full detail
            
            if (this.chart.data.datasets[0].data.length > maxVisiblePoints) {
                // Compress older data by keeping every Nth point
                const compressionRatio = Math.ceil(this.chart.data.datasets[0].data.length / maxVisiblePoints);
                
                // Keep recent episodes at full resolution, compress older ones
                const recentPoints = 50; // Keep last 50 episodes at full resolution
                const fullResolutionData = this.chart.data.datasets[0].data.slice(-recentPoints);
                const fullResolutionAvg = this.chart.data.datasets[1].data.slice(-recentPoints);
                
                // Compress older data
                const olderData = this.chart.data.datasets[0].data.slice(0, -recentPoints);
                const olderAvgData = this.chart.data.datasets[1].data.slice(0, -recentPoints);
                
                const compressedData = [];
                const compressedAvgData = [];
                
                for (let i = 0; i < olderData.length; i += compressionRatio) {
                    if (olderData[i]) {
                        compressedData.push(olderData[i]);
                    }
                    if (olderAvgData[i]) {
                        compressedAvgData.push(olderAvgData[i]);
                    }
                }
                
                // Combine compressed older data with full resolution recent data
                this.chart.data.datasets[0].data = [...compressedData, ...fullResolutionData];
                this.chart.data.datasets[1].data = [...compressedAvgData, ...fullResolutionAvg];
                
                // Update labels to match
                this.chart.data.labels = this.chart.data.datasets[0].data.map(point => point.x);
            }

            // Update chart options for better scaling
            this.chart.options.scales.x.min = Math.max(1, Math.min(...this.chart.data.labels) - 5);
            this.chart.options.scales.x.max = Math.max(...this.chart.data.labels) + 5;
            
            // Set appropriate tick intervals based on data range
            const dataRange = this.chart.options.scales.x.max - this.chart.options.scales.x.min;
            this.chart.options.scales.x.ticks.stepSize = Math.max(1, Math.floor(dataRange / 10));

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
        
        // Add CSV export button event listener
        document.getElementById('exportButton').addEventListener('click', () => {
            console.log('Exporting CSV report...');
            this.exportToCSV();
        });
        
        // Add Q-table export button event listener
        document.getElementById('exportQTableButton').addEventListener('click', () => {
            console.log('Exporting Q-table...');
            this.exportQTable();
        });
        
        // Add Monte Carlo export button event listener
        document.getElementById('exportMonteCarloButton').addEventListener('click', () => {
            console.log('Exporting Monte Carlo tables...');
            this.exportMonteCarloTables();
        });
        
        // Add Actor-Critic export button event listener
        document.getElementById('exportActorCriticButton').addEventListener('click', () => {
            console.log('Exporting Actor-Critic tables...');
            this.exportActorCriticTables();
        });
        
        // Add episode-specific export button event listener
        document.getElementById('exportSpecificEpisodeButton').addEventListener('click', () => {
            console.log('Exporting specific episode data...');
            this.exportSpecificEpisodeData();
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
        
        // Reset all metrics
        this.episodeMetrics = [];
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

            // Calculate and display success rate
            const successRate = this.episodeCount > 0 ? 
                (this.env.successfulCompletions / this.episodeCount * 100) : 0;
            document.getElementById('successRate').textContent = successRate.toFixed(1) + '%';

            // Update best and average rewards - fix NaN issue
            const allRewards = this.chart.data.datasets[0].data;
            if (allRewards.length > 0) {
                // Extract reward values from chart data (handle both number and {x,y} object formats)
                const rewardValues = allRewards.map(item => 
                    typeof item === 'object' ? item.y : item
                ).filter(val => !isNaN(val) && val !== undefined);
                
                if (rewardValues.length > 0) {
                    const bestReward = Math.max(...rewardValues);
                    const recentRewards = rewardValues.slice(-10);
                    const avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
                    
                    document.getElementById('bestReward').textContent = bestReward.toFixed(1);
                    document.getElementById('averageReward').textContent = avgReward.toFixed(1);
                } else {
                    document.getElementById('bestReward').textContent = '0.0';
                    document.getElementById('averageReward').textContent = '0.0';
                }
            } else {
                document.getElementById('bestReward').textContent = '0.0';
                document.getElementById('averageReward').textContent = '0.0';
            }

            // Update top episodes display
            this.updateTopEpisodes();

            // Trigger a render to update heatmap
            this.env.render();
        } catch (error) {
            console.error('Error updating status panel:', error);
        }
    }

    // New method to track and display top episodes
    updateTopEpisodes() {
        if (this.episodeMetrics.length === 0) {
            document.getElementById('topEpisodes').textContent = 'None';
            return;
        }

        // Get top 3 episodes by reward
        const topEpisodes = [...this.episodeMetrics]
            .sort((a, b) => b.currentReward - a.currentReward)
            .slice(0, 3);

        // Format the display string
        let displayText = '';
        topEpisodes.forEach((episode, index) => {
            if (index > 0) displayText += ', ';
            displayText += `#${episode.episode} (${episode.currentReward.toFixed(1)})`;
        });

        // If there are ties (same reward), show them
        const bestReward = topEpisodes[0].currentReward;
        const tiedEpisodes = this.episodeMetrics.filter(ep => 
            Math.abs(ep.currentReward - bestReward) < 0.001
        );

        if (tiedEpisodes.length > 1) {
            const tiedNumbers = tiedEpisodes.map(ep => `#${ep.episode}`).join(', ');
            displayText = `${tiedNumbers} (${bestReward.toFixed(1)}) [${tiedEpisodes.length} tied]`;
        }

        document.getElementById('topEpisodes').textContent = displayText;
    }

    // CSV Export functionality
    exportToCSV() {
        if (this.episodeMetrics.length === 0) {
            alert('No training data available to export. Please run some episodes first.');
            return;
        }

        const headers = [
            'Episode',
            'Algorithm',
            'Current Reward',
            'Average Reward (Last 10)',
            'Steps to Complete',
            'Episode Time (s)',
            'Trash Remaining',
            'Initial Trash Count',
            'Trash Collected',
            'Is Successful Completion',
            'Total Successful Completions',
            'Learning Rate',
            'Epsilon',
            'Gamma',
            'Evil Robot Enabled',
            'Max Steps',
            'Timestamp'
        ];

        // Create rows without the redundant episode marker columns
        const dataRows = this.episodeMetrics.map((metric) => {
            const row = [
                metric.episode,
                metric.algorithm,
                metric.currentReward.toFixed(2),
                metric.averageReward.toFixed(2),
                metric.stepsToComplete,
                metric.episodeTime.toFixed(2),
                metric.trashRemaining,
                metric.initialTrashCount,
                metric.trashCollected,
                metric.isSuccessfulCompletion,
                metric.totalSuccessfulCompletions,
                metric.learningRate,
                metric.epsilon,
                metric.gamma,
                metric.evilRobotEnabled,
                metric.maxSteps,
                metric.timestamp
            ];
            
            return row.join(',');
        });

        const csvContent = [
            `# REINFORCEMENT LEARNING TRAINING REPORT`,
            `# Generated: ${new Date().toISOString()}`,
            `# Total episodes in this export: ${this.episodeMetrics.length}`,
            `#`,
            headers.join(','),
            ...dataRows
        ].join('\n');

        // Create and download the CSV file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        
        // Generate filename with timestamp and algorithm
        const algorithm = document.getElementById('algorithm').value;
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
        const filename = `RL_Training_Report_${algorithm}_${timestamp}.csv`;
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        console.log(`✅ Exported ${this.episodeMetrics.length} episodes to CSV: ${filename}`);
        alert(`Successfully exported ${this.episodeMetrics.length} episodes to CSV!\n\nFilename: ${filename}`);
    }

    // Q-table Export functionality (Q-Learning only)
    exportQTable() {
        const algorithm = document.getElementById('algorithm').value;
        
        if (algorithm !== 'q_learning') {
            alert('Q-table export is only available for Q-Learning algorithm. Please switch to Q-Learning and train some episodes first.');
            return;
        }

        if (!this.agent || !this.agent.qTable || Object.keys(this.agent.qTable).length === 0) {
            alert('No Q-table data available to export. Please run some Q-Learning episodes first.');
            return;
        }

        try {
            // Get Q-table data and statistics
            const qtableData = this.agent.exportQTable();
            const stats = this.agent.getQTableStats();

            if (qtableData.length === 0) {
                alert('Q-table is empty. Please run some episodes first.');
                return;
            }

            // Create CSV headers
            const headers = [
                'State',
                'Wall-E X',
                'Wall-E Y', 
                'Evil Robot X',
                'Evil Robot Y',
                'Trash Count',
                'Action ID',
                'Action Name',
                'Q-Value',
                'Is Optimal Action'
            ];

            // Create CSV content
            const csvContent = [
                `# Q-LEARNING Q-TABLE EXPORT`,
                `# Generated: ${new Date().toISOString()}`,
                `# Algorithm: Q-Learning`,
                `# Total States: ${stats.totalStates}`,
                `# Total Entries: ${stats.totalEntries}`,
                `# Non-Zero Entries: ${stats.nonZeroEntries}`,
                `# Coverage: ${stats.coverage}`,
                `# Q-Value Range: ${stats.minQValue} to ${stats.maxQValue}`,
                `#`,
                headers.join(','),
                ...qtableData.map(row => [
                    `"${row.state}"`,          
                    row.wallE_x,
                    row.wallE_y,
                    row.evil_x,
                    row.evil_y,
                    row.trashCount,
                    row.action,
                    `"${row.actionName}"`,     
                    row.qValue,
                    row.isOptimalAction
                ].join(','))
            ].join('\n');

            // Download the CSV file
            const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
            this.downloadCSV(csvContent, `Q_Table_Export_${timestamp}.csv`);
            
            console.log(`Exported Q-table with ${qtableData.length} entries`);
            alert(`🧠 Q-TABLE SUCCESSFULLY EXPORTED! 🧠\n\n📊 STATISTICS:\n• States: ${stats.totalStates}\n• Q-Values: ${stats.totalEntries}\n• Coverage: ${stats.coverage}\n• Value Range: ${stats.minQValue} to ${stats.maxQValue}`);
            
        } catch (error) {
            console.error('Error exporting Q-table:', error);
            alert('Error exporting Q-table. Please check the console for details.');
        }
    }

    // Monte Carlo Return Table Export functionality
    exportMonteCarloTables() {
        const algorithm = document.getElementById('algorithm').value;
        
        if (algorithm !== 'monte_carlo') {
            alert('Return table export is only available for Monte Carlo algorithm. Please switch to Monte Carlo and train some episodes first.');
            return;
        }

        if (!this.agent || !this.agent.qTable || Object.keys(this.agent.qTable).length === 0) {
            alert('No return table data available to export. Please run some Monte Carlo episodes first.');
            return;
        }

        try {
            // Get return table data and statistics
            const returnData = this.agent.exportReturnTable();
            const policyData = this.agent.exportPolicyTable();
            const stats = this.agent.getReturnTableStats();

            if (returnData.length === 0) {
                alert('Return table is empty. Please run some episodes first.');
                return;
            }

            // Export Return Table
            const returnHeaders = [
                'State', 'Wall-E X', 'Wall-E Y', 'Evil Robot X', 'Evil Robot Y', 
                'Trash Count', 'Action ID', 'Action Name', 'Return Value', 'Is Optimal Action'
            ];

            const returnCSV = [
                `# MONTE CARLO RETURN TABLE EXPORT`,
                `# Generated: ${new Date().toISOString()}`,
                `# Algorithm: Monte Carlo`,
                `# Total States: ${stats.totalStates}`,
                `# Total Entries: ${stats.totalEntries}`,
                `# Non-Zero Entries: ${stats.nonZeroEntries}`,
                `# Coverage: ${stats.coverage}`,
                `# Return Range: ${stats.minReturn} to ${stats.maxReturn}`,
                `# Average Return: ${stats.avgReturn}`,
                `#`,
                returnHeaders.join(','),
                ...returnData.map(row => [
                    `"${row.state}"`, row.wallE_x, row.wallE_y, row.evil_x, row.evil_y,
                    row.trashCount, row.action, `"${row.actionName}"`, row.returnValue, row.isOptimalAction
                ].join(','))
            ].join('\n');

            // Export Policy Table
            const policyHeaders = [
                'State', 'Wall-E X', 'Wall-E Y', 'Evil Robot X', 'Evil Robot Y',
                'Trash Count', 'Optimal Action ID', 'Optimal Action Name', 'Return Value'
            ];

            const policyCSV = [
                `# MONTE CARLO POLICY TABLE EXPORT`,
                `# Generated: ${new Date().toISOString()}`,
                `# Algorithm: Monte Carlo`,
                `# Total States: ${stats.totalStates}`,
                `# This shows the optimal action for each state`,
                `#`,
                policyHeaders.join(','),
                ...policyData.map(row => [
                    `"${row.state}"`, row.wallE_x, row.wallE_y, row.evil_x, row.evil_y,
                    row.trashCount, row.optimalAction, `"${row.optimalActionName}"`, row.returnValue
                ].join(','))
            ].join('\n');

            // Download both tables
            const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
            this.downloadCSV(returnCSV, `MC_Return_Table_${timestamp}.csv`);
            
            setTimeout(() => {
                this.downloadCSV(policyCSV, `MC_Policy_Table_${timestamp}.csv`);
            }, 500);
            
            console.log(`Exported Monte Carlo tables with ${returnData.length} return entries and ${policyData.length} policy entries`);
            alert(`🧠 MONTE CARLO TABLES EXPORTED! 🧠\n\n📊 STATISTICS:\n• States: ${stats.totalStates}\n• Return Entries: ${stats.totalEntries}\n• Coverage: ${stats.coverage}\n• Return Range: ${stats.minReturn} to ${stats.maxReturn}\n\nTwo files downloaded:\n1. Return Table (all state-action returns)\n2. Policy Table (optimal actions only)`);
            
        } catch (error) {
            console.error('Error exporting Monte Carlo tables:', error);
            alert('Error exporting Monte Carlo tables. Please check the console for details.');
        }
    }

    // Actor-Critic Value and Policy Tables Export functionality
    exportActorCriticTables() {
        const algorithm = document.getElementById('algorithm').value;
        
        if (algorithm !== 'actor_critic') {
            alert('Value/Policy table export is only available for Actor-Critic algorithm. Please switch to Actor-Critic and train some episodes first.');
            return;
        }

        if (!this.agent) {
            alert('No Actor-Critic agent available. Please run some Actor-Critic episodes first.');
            return;
        }

        try {
            // Get value and policy table data
            const valueData = this.agent.exportValueTable();
            const policyData = this.agent.exportPolicyTable();
            const stats = this.agent.getActorCriticStats();

            if (valueData.length === 0 || policyData.length === 0) {
                alert('Actor-Critic tables are empty. Please run some episodes first.');
                return;
            }

            // Export Value Table
            const valueHeaders = [
                'State', 'Wall-E X', 'Wall-E Y', 'Evil Robot X', 'Evil Robot Y',
                'Trash Count', 'State Value'
            ];

            const valueCSV = [
                `# ACTOR-CRITIC VALUE TABLE EXPORT`,
                `# Generated: ${new Date().toISOString()}`,
                `# Algorithm: Actor-Critic`,
                `# Network Architecture: ${stats.networkArchitecture}`,
                `# Temperature: ${stats.temperature}`,
                `# Sample Value Range: ${stats.minSampleValue} to ${stats.maxSampleValue}`,
                `# Average Sample Value: ${stats.avgSampleValue}`,
                `# Total States: ${stats.totalStates}`,
                `#`,
                valueHeaders.join(','),
                ...valueData.map(row => [
                    `"${row.state}"`, row.wallE_x, row.wallE_y, row.evil_x, row.evil_y,
                    row.trashCount, row.stateValue
                ].join(','))
            ].join('\n');

            // Export Policy Table
            const policyHeaders = [
                'State', 'Wall-E X', 'Wall-E Y', 'Evil Robot X', 'Evil Robot Y',
                'Trash Count', 'Optimal Action ID', 'Optimal Action Name',
                'Prob Up', 'Prob Right', 'Prob Down', 'Prob Left', 'Max Probability'
            ];

            const policyCSV = [
                `# ACTOR-CRITIC POLICY TABLE EXPORT`,
                `# Generated: ${new Date().toISOString()}`,
                `# Algorithm: Actor-Critic`,
                `# Network Architecture: ${stats.networkArchitecture}`,
                `# Temperature: ${stats.temperature}`,
                `# This shows action probabilities for each state`,
                `#`,
                policyHeaders.join(','),
                ...policyData.map(row => [
                    `"${row.state}"`, row.wallE_x, row.wallE_y, row.evil_x, row.evil_y,
                    row.trashCount, row.optimalAction, `"${row.optimalActionName}"`,
                    row.actionProbUp, row.actionProbRight, row.actionProbDown, 
                    row.actionProbLeft, row.maxProbability
                ].join(','))
            ].join('\n');

            // Download both tables
            const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
            this.downloadCSV(valueCSV, `AC_Value_Table_${timestamp}.csv`);
            
            setTimeout(() => {
                this.downloadCSV(policyCSV, `AC_Policy_Table_${timestamp}.csv`);
            }, 500);
            
            console.log(`Exported Actor-Critic tables with ${valueData.length} value entries and ${policyData.length} policy entries`);
            alert(`🎭 ACTOR-CRITIC TABLES EXPORTED! 🎭\n\n📊 STATISTICS:\n• Network: ${stats.networkArchitecture}\n• Temperature: ${stats.temperature}\n• Total States: ${stats.totalStates}\n• Sample Value Range: ${stats.minSampleValue} to ${stats.maxSampleValue}\n\nTwo files downloaded:\n1. Value Table (state values from critic)\n2. Policy Table (action probabilities from actor)`);
            
        } catch (error) {
            console.error('Error exporting Actor-Critic tables:', error);
            alert('Error exporting Actor-Critic tables. Please check the console for details.');
        }
    }

    // Helper method to download CSV files
    downloadCSV(csvContent, filename) {
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // Episode-Specific Export functionality for all algorithms
    exportSpecificEpisodeData() {
        const episodeNumber = parseInt(document.getElementById('specificEpisodeNumber').value);
        const algorithm = document.getElementById('algorithm').value;
        
        // Validate episode number
        if (!episodeNumber || episodeNumber < 1) {
            alert('Please enter a valid episode number (1 or greater).');
            return;
        }
        
        if (!this.agent) {
            alert('No agent available. Please run some training episodes first.');
            return;
        }

        console.log(`Exporting data for episode ${episodeNumber} (${algorithm})`);

        try {
            let exportData = [];
            let filename = '';
            let csvContent = '';
            
            switch (algorithm) {
                case 'q_learning':
                    exportData = this.exportQLearningEpisodeData(episodeNumber);
                    if (exportData.length === 0) {
                        alert(`No Q-Learning data found for episode ${episodeNumber}. Make sure the episode exists and has been completed.`);
                        return;
                    }
                    filename = `Q_Learning_Episode_${episodeNumber}_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.csv`;
                    csvContent = this.createQLearningEpisodeCSV(exportData, episodeNumber);
                    break;
                    
                case 'monte_carlo':
                    exportData = this.exportMonteCarloEpisodeData(episodeNumber);
                    if (exportData.length === 0) {
                        alert(`No Monte Carlo data found for episode ${episodeNumber}. Make sure the episode exists and has been completed.`);
                        return;
                    }
                    filename = `Monte_Carlo_Episode_${episodeNumber}_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.csv`;
                    csvContent = this.createMonteCarloEpisodeCSV(exportData, episodeNumber);
                    break;
                    
                case 'actor_critic':
                    exportData = this.exportActorCriticEpisodeData(episodeNumber);
                    if (exportData.length === 0) {
                        alert(`No Actor-Critic data found for episode ${episodeNumber}. Make sure the episode exists and has been completed.`);
                        return;
                    }
                    filename = `Actor_Critic_Episode_${episodeNumber}_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.csv`;
                    csvContent = this.createActorCriticEpisodeCSV(exportData, episodeNumber);
                    break;
                    
                default:
                    alert('Unknown algorithm selected.');
                    return;
            }
            
            // Download the CSV file
            this.downloadCSV(csvContent, filename);
            
            console.log(`Successfully exported episode ${episodeNumber} data for ${algorithm}`);
            alert(`🎯 EPISODE ${episodeNumber} DATA EXPORTED! 🎯\n\nAlgorithm: ${algorithm.toUpperCase()}\nData entries: ${exportData.length}\nFilename: ${filename}`);
            
        } catch (error) {
            console.error('Error exporting specific episode data:', error);
            alert('Error exporting episode data. Please check the console for details.');
        }
    }

    // Q-Learning episode-specific export
    exportQLearningEpisodeData(episodeNumber) {
        if (!this.agent.tableHistory) return [];
        
        // Get all entries from the specified episode (not just reset markers)
        return this.agent.tableHistory.filter(entry => {
            return entry.episode === episodeNumber;
        });
    }

    // Monte Carlo episode-specific export
    exportMonteCarloEpisodeData(episodeNumber) {
        if (!this.agent.returnHistory) {
            console.log('No returnHistory found');
            return [];
        }
        
        console.log(`Looking for Monte Carlo episode ${episodeNumber}`);
        console.log(`Total returnHistory entries: ${this.agent.returnHistory.length}`);
        
        // Get all entries from the specified episode (including reset markers and regular data)
        const episodeData = this.agent.returnHistory.filter(entry => {
            return entry.episode === episodeNumber && entry.episodeReset === 'FALSE';
        });
        
        console.log(`Found ${episodeData.length} Monte Carlo entries for episode ${episodeNumber}`);
        
        // Debug: Show first few entries
        if (episodeData.length > 0) {
            console.log('Sample entries:', episodeData.slice(0, 3));
        }
        
        return episodeData;
    }

    // Actor-Critic episode-specific export
    exportActorCriticEpisodeData(episodeNumber) {
        if (!this.agent.updateHistory) {
            console.log('No updateHistory found');
            return [];
        }
        
        console.log(`Looking for Actor-Critic episode ${episodeNumber}`);
        console.log(`Total updateHistory entries: ${this.agent.updateHistory.length}`);
        
        // Get all entries from the specified episode (including reset markers and regular data)
        const episodeData = this.agent.updateHistory.filter(entry => {
            return entry.episode === episodeNumber && entry.episodeReset === 'FALSE';
        });
        
        console.log(`Found ${episodeData.length} Actor-Critic entries for episode ${episodeNumber}`);
        
        // Debug: Show first few entries
        if (episodeData.length > 0) {
            console.log('Sample entries:', episodeData.slice(0, 3));
        }
        
        return episodeData;
    }

    // Create CSV content for Q-Learning episode data
    createQLearningEpisodeCSV(data, episodeNumber) {
        const headers = [
            'Step', 'State', 'Wall-E X', 'Wall-E Y', 'Evil Robot X', 'Evil Robot Y',
            'Trash Count', 'Action ID', 'Action Name', 'Q-Value', 'Is Optimal Action'
        ];

        const csvContent = [
            `# Q-LEARNING EPISODE ${episodeNumber} DATA EXPORT`,
            `# Generated: ${new Date().toISOString()}`,
            `# Algorithm: Q-Learning`,
            `# Episode: ${episodeNumber}`,
            `# Total steps in episode: ${data.length}`,
            `#`,
            headers.join(','),
            ...data.map((entry, index) => [
                index + 1,
                `"${entry.state}"`,
                entry.wallEX,
                entry.wallEY,
                entry.evilRobotX,
                entry.evilRobotY,
                entry.trashCount,
                entry.action,
                `"${entry.actionName}"`,
                entry.qValue.toFixed(4),
                entry.isOptimal ? 'TRUE' : 'FALSE'
            ].join(','))
        ].join('\n');

        return csvContent;
    }

    // Create CSV content for Monte Carlo episode data
    createMonteCarloEpisodeCSV(data, episodeNumber) {
        const headers = [
            'Step', 'State', 'Action ID', 'Return Value', 'Reward', 'Timestamp'
        ];

        const csvContent = [
            `# MONTE CARLO EPISODE ${episodeNumber} DATA EXPORT`,
            `# Generated: ${new Date().toISOString()}`,
            `# Algorithm: Monte Carlo`,
            `# Episode: ${episodeNumber}`,
            `# Total steps in episode: ${data.length}`,
            `#`,
            headers.join(','),
            ...data.map(entry => [
                entry.step,
                `"${entry.state}"`,
                entry.action,
                entry.returnValue,
                entry.reward,
                entry.timestamp
            ].join(','))
        ].join('\n');

        return csvContent;
    }

    // Create CSV content for Actor-Critic episode data
    createActorCriticEpisodeCSV(data, episodeNumber) {
        const headers = [
            'Step', 'State', 'Action ID', 'State Value', 'Reward', 'Timestamp'
        ];

        const csvContent = [
            `# ACTOR-CRITIC EPISODE ${episodeNumber} DATA EXPORT`,
            `# Generated: ${new Date().toISOString()}`,
            `# Algorithm: Actor-Critic`,
            `# Episode: ${episodeNumber}`,
            `# Total steps in episode: ${data.length}`,
            `#`,
            headers.join(','),
            ...data.map(entry => [
                entry.step,
                `"${entry.state}"`,
                entry.action,
                entry.stateValue,
                entry.reward,
                entry.timestamp
            ].join(','))
        ].join('\n');

        return csvContent;
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
            
            // 🚀 EPISODE START MARKER - ADD VISUAL INDICATOR
            console.log(`\n🔄 ===== EPISODE ${this.episodeCount} START =====`);
            console.log(`📍 Environment Reset: Wall-E back to start position`);
            console.log(`🎯 Algorithm: ${document.getElementById('algorithm').value.toUpperCase()}`);
            console.log(`🗑️ Initial Trash Count: ${this.env.trashCount}`);
            
            // ADD VISUAL EPISODE START INDICATOR
            const statusElement = document.getElementById('episodeCounter');
            statusElement.style.backgroundColor = '#4CAF50';
            statusElement.style.color = 'white';
            statusElement.style.fontWeight = 'bold';
            setTimeout(() => {
                statusElement.style.backgroundColor = '';
                statusElement.style.color = '';
                statusElement.style.fontWeight = '';
            }, 1000);
            
            // 🎯 NOTIFY AGENT OF NEW EPISODE FOR TABLE TRACKING
            if (this.agent.setEpisode) {
                this.agent.setEpisode(this.episodeCount);
            }
            
            let state = this.env.reset();
            this.totalReward = 0;
            let done = false;
            
            // Track episode metrics
            this.currentEpisodeSteps = 0;
            this.episodeStartTime = Date.now();
            const initialTrashCount = this.env.trashCount;
            let previousSuccessCount = this.env.successfulCompletions;

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
                    this.currentEpisodeSteps++;
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

            // Record episode metrics
            const episodeTime = (Date.now() - this.episodeStartTime) / 1000;
            const isSuccessful = this.env.successfulCompletions > previousSuccessCount;
            
            // 🏁 EPISODE END MARKER - ADD VISUAL INDICATOR
            console.log(`\n🏁 ===== EPISODE ${this.episodeCount} END =====`);
            console.log(`⏱️ Duration: ${episodeTime.toFixed(2)}s | Steps: ${this.currentEpisodeSteps}`);
            console.log(`💰 Total Reward: ${this.totalReward.toFixed(2)}`);
            console.log(`🗑️ Trash Collected: ${initialTrashCount - this.env.trashCount}/${initialTrashCount}`);
            console.log(`${isSuccessful ? '✅ SUCCESS!' : '❌ Failed'} | Overall Success Rate: ${((this.env.successfulCompletions / this.episodeCount) * 100).toFixed(1)}%`);
            
            // ADD VISUAL EPISODE END INDICATOR
            const rewardElement = document.getElementById('currentReward');
            rewardElement.style.backgroundColor = isSuccessful ? '#4CAF50' : '#f44336';
            rewardElement.style.color = 'white';
            rewardElement.style.fontWeight = 'bold';
            setTimeout(() => {
                rewardElement.style.backgroundColor = '';
                rewardElement.style.color = '';
                rewardElement.style.fontWeight = '';
            }, 1500);
            
            // Fix average reward calculation for CSV export
            let avgReward = this.totalReward; // Default to current reward for first episode
            if (this.episodeMetrics.length > 0) {
                // Get the last 10 episode rewards from stored metrics
                const recentMetrics = this.episodeMetrics.slice(-9); // Get last 9 episodes
                const recentRewards = recentMetrics.map(m => m.currentReward).concat([this.totalReward]); // Add current
                avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
            }
            
            const episodeMetric = {
                episode: this.episodeCount,
                algorithm: document.getElementById('algorithm').value,
                currentReward: this.totalReward,
                averageReward: avgReward,
                stepsToComplete: this.currentEpisodeSteps,
                episodeTime: episodeTime,
                trashRemaining: this.env.trashCount,
                initialTrashCount: initialTrashCount,
                trashCollected: initialTrashCount - this.env.trashCount,
                isSuccessfulCompletion: isSuccessful,
                totalSuccessfulCompletions: this.env.successfulCompletions,
                learningRate: this.getParams().learningRate || this.getParams().actorLearningRate || 'N/A',
                epsilon: this.getParams().epsilon || 'N/A',
                gamma: this.getParams().gamma,
                evilRobotEnabled: document.getElementById('evilRobotEnabled').checked,
                maxSteps: this.env.maxSteps,
                timestamp: new Date().toISOString()
            };
            
            this.episodeMetrics.push(episodeMetric);

            // Show episode transition notification on screen
            this.showEpisodeNotification(this.episodeCount, isSuccessful, this.totalReward);

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
                console.log(`\n🎯 ===== TRAINING COMPLETE =====`);
                console.log(`📊 Total Episodes: ${this.episodeCount}`);
                console.log(`✅ Successful Completions: ${this.env.successfulCompletions}`);
                console.log(`📈 Final Success Rate: ${((this.env.successfulCompletions / this.episodeCount) * 100).toFixed(1)}%`);
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

    // Add visual episode notification method
    showEpisodeNotification(episodeNum, isSuccessful, reward) {
        // Create or get notification element
        let notification = document.getElementById('episodeNotification');
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'episodeNotification';
            notification.style.cssText = `
                position: fixed;
                top: 10px;
                right: 10px;
                padding: 10px 15px;
                border-radius: 5px;
                font-weight: bold;
                z-index: 1000;
                transition: opacity 0.3s ease;
                max-width: 300px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            `;
            document.body.appendChild(notification);
        }

        // Set notification content and style
        const bgColor = isSuccessful ? '#4CAF50' : '#f44336';
        const status = isSuccessful ? '✅ SUCCESS' : '❌ FAILED';
        
        notification.style.backgroundColor = bgColor;
        notification.style.color = 'white';
        notification.style.opacity = '1';
        notification.innerHTML = `
            🔄 <strong>Episode ${episodeNum} Complete</strong><br>
            ${status} | Reward: ${reward.toFixed(1)}
        `;

        // Auto-hide after 2 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
        }, 2000);
    }
}

// Start simulation when page loads
window.addEventListener('load', () => {
    const simulation = new Simulation();
});