# Pokemon RL Project Setup Plan

## Phase 1: Repository Setup (Day 1)

### Create New Repository
```bash
# On GitHub, create new repo: pokemon-rl-deployment
# Description: "Production deployment of Pokemon Red RL agent on AWS EC2"
# Initialize with: README, .gitignore (Python), MIT License

# Clone locally
git clone https://github.com/yourusername/pokemon-rl-deployment.git
cd pokemon-rl-deployment
```

### Initial Project Structure
```bash
# Create directory structure
mkdir -p {configs,scripts,src/{agent,env,training,inference},notebooks,tests,terraform,docs,assets}

# Create placeholder files
touch requirements.txt
touch .env.example
touch docker-compose.yml
touch Dockerfile
```

### Copy the README
```bash
# Use the README.md I just created
# Update with your actual information
```

## Phase 2: Research & Setup (Days 1-3)

### Study the Modern Implementation
1. **Read the website thoroughly**: https://drubinstein.github.io/pokerl/
2. **Read the arXiv paper**: https://arxiv.org/abs/2502.19920
3. **Clone the reference implementation**:
   ```bash
   # In a separate directory (not your repo yet)
   git clone https://github.com/drubinstein/pokemonred_puffer.git
   cd pokemonred_puffer
   ```
4. **Join the Discord**: http://discord.gg/RvadteZk4G
5. **Read through issues/discussions** to understand common problems

### Local Testing Setup
```bash
# Create Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install pokemonred_puffer dependencies
cd pokemonred_puffer
pip install -r requirements.txt

# Get Pokemon Red ROM (legally!)
# Place it in the appropriate directory

# Run pretrained model to verify setup
python run_pretrained.py  # or whatever their command is
```

### Document Everything
- Take notes on setup issues
- Screenshot any errors and solutions
- Note dependency versions that work
- Track time spent (good for Medium article)

## Phase 3: Understand the Codebase (Days 4-7)

### Code Reading Checklist
- [ ] How does the environment wrapper work?
- [ ] What's the reward structure?
- [ ] How is observation space defined?
- [ ] How does checkpointing work?
- [ ] What hyperparameters are configurable?
- [ ] How does distributed training work?
- [ ] What metrics are tracked?

### Create Analysis Notebooks
```python
# notebooks/01_environment_exploration.ipynb
# - Load environment
# - Visualize observations
# - Test action space
# - Understand reward function

# notebooks/02_pretrained_analysis.ipynb
# - Load pretrained model
# - Analyze behavior
# - Test different game states
# - Extract learned features
```

## Phase 4: AWS Setup (Days 7-10)

### AWS Account Prep
1. Create/verify AWS account
2. Set up billing alerts (critical!)
   - Alert at $25, $50, $75
3. Create IAM user with appropriate permissions
4. Generate access keys
5. Configure AWS CLI locally

### Infrastructure Planning
```bash
# terraform/main.tf - Basic EC2 setup
# - VPC configuration
# - Security groups (SSH, TensorBoard ports)
# - EC2 instance definition
# - S3 bucket for checkpoints
# - CloudWatch log group

# scripts/deploy_ec2.sh
# - Launch instance
# - Install dependencies
# - Configure environment
# - Start training

# scripts/setup_monitoring.sh
# - Configure CloudWatch metrics
# - Set up TensorBoard
# - Create dashboards
```

### Test on Small Instance First
- Launch t3.medium (~$0.04/hr)
- Verify everything works
- Test checkpoint/restore
- Validate monitoring
- Kill instance (don't forget!)

## Phase 5: Initial Training Run (Days 10-14)

### Pre-flight Checklist
- [ ] All scripts tested locally
- [ ] Billing alerts active
- [ ] Checkpoint mechanism verified
- [ ] Monitoring dashboard created
- [ ] tmux/screen setup for persistence
- [ ] Auto-shutdown script configured

### Launch Training
```bash
# Use spot instance for cost savings
./scripts/deploy_ec2.sh \
  --instance-type c5.2xlarge \
  --spot \
  --max-price 0.15

# SSH in and start
ssh -i ~/.ssh/pokemon-rl.pem ubuntu@<ip>
cd pokemon-rl-deployment
./scripts/start_training.sh --config configs/baseline.yaml

# Detach from tmux: Ctrl+B, then D
```

### Monitor Progress
- Check TensorBoard daily
- Review CloudWatch metrics
- Verify checkpoints saving to S3
- Track costs in AWS billing dashboard

### Expected Timeline for First Gym
- ~24-48 hours of training
- ~500K-1M episodes
- Cost: $15-30 on spot instances

## Phase 6: Documentation & Iteration (Days 14-21)

### Document Results
- Export TensorBoard data
- Create training curve visualizations
- Calculate total costs
- Note any issues/challenges

### Create Demo Assets
```bash
# Record gameplay video
python scripts/record_gameplay.py --model checkpoints/best_model.pt

# Create GIF for README
ffmpeg -i gameplay.mp4 -vf "fps=10,scale=320:-1" assets/demo.gif

# Export metrics
python scripts/export_metrics.py --output assets/training_curve.png
```

### Update Repository
- Update README with actual results
- Add screenshots and GIFs
- Include cost breakdown
- Write detailed setup instructions
- Add troubleshooting section

## Phase 7: Advanced Features (Days 21-30)

### Build Inference API
```python
# src/inference/api.py - FastAPI server
# - Load trained model
# - Accept game state
# - Return action
# - Stream gameplay video

# Docker containerization
# - Dockerfile for API
# - docker-compose for local testing
```

### Web Interface (Optional)
```bash
# Simple React frontend
# - Display game screen
# - Show agent's reasoning
# - Control playback
# - Display statistics
```

### Write Medium Article
**Title**: "Deploying Reinforcement Learning at Scale: Pokemon Red on AWS"

**Outline**:
1. Introduction - The challenge
2. Why Pokemon Red matters for RL
3. Architecture decisions
4. Cost optimization strategies
5. Challenges encountered
6. Results and learnings
7. Future work

## Phase 8: Polish & Promotion (Days 30+)

### Repository Polish
- [ ] Clean commit history
- [ ] Add comprehensive tests
- [ ] Create CONTRIBUTING.md
- [ ] Add CI/CD pipeline
- [ ] Create detailed docs/
- [ ] Add code comments

### Promotion
- [ ] Post on LinkedIn with results
- [ ] Share in relevant subreddits (r/MachineLearning, r/reinforcementlearning)
- [ ] Tweet about it (tag @PufferAI, authors)
- [ ] Share in Pokemon RL Discord
- [ ] Add to portfolio website
- [ ] Update resume

### Job Application Integration
When applying, mention:
- "Recently completed production deployment of cutting-edge RL system"
- "Optimized AWS infrastructure, reducing costs 66% vs. on-demand"
- "Implemented distributed training with automatic fault recovery"
- Link to repo in portfolio section

## Budget Planning

### Minimum Viable ($50-75)
- Testing: $10 (small instances)
- One training run: $30 (spot c5.2xlarge, 48hrs)
- Storage: $5 (S3 + snapshots)
- Buffer: $10

### Recommended ($100-150)
- Testing: $15
- Two training runs: $60 (iterations/improvements)
- Inference API testing: $10
- Storage: $10
- Buffer: $20

### Cost Control Measures
1. **Always use spot instances** for training
2. **Set strict billing alarms**
3. **Stop (don't terminate) instances** between sessions
4. **Delete old checkpoints** from S3
5. **Use t3.micro** for API server (free tier eligible)
6. **Test locally first** before cloud deployment

## Success Metrics

### Technical
- [ ] Successfully train agent to first gym
- [ ] Implement checkpoint/restore
- [ ] Set up monitoring pipeline
- [ ] Deploy inference API
- [ ] Stay under budget

### Portfolio
- [ ] README with results
- [ ] Medium article published
- [ ] 3+ demo assets (GIF, graphs, etc.)
- [ ] Clean, documented code
- [ ] LinkedIn post with engagement

### Career
- [ ] Include in 5+ job applications
- [ ] Discuss in 2+ interviews
- [ ] Use to demonstrate AI Engineering skills
- [ ] Build confidence in cloud infrastructure

## Risk Mitigation

### Technical Risks
- **Training doesn't converge**: Start with proven hyperparameters
- **AWS costs spiral**: Billing alarms + spot instances
- **Instance interruptions**: Robust checkpointing
- **Dependency hell**: Document exact versions

### Timeline Risks
- **Taking too long**: Set hard deadlines per phase
- **Scope creep**: Focus on MVP first, iterate later
- **Analysis paralysis**: Just start! Learn by doing

## Next Immediate Steps

1. **Today**: Delete old fork, create new repo
2. **Tomorrow**: Clone pokemonred_puffer, get it running locally
3. **This week**: Understand codebase, join Discord, read paper
4. **Next week**: AWS setup, first training run
5. **By end of month**: Complete project, publish Medium article

Remember: **Done is better than perfect.** You can always iterate!
