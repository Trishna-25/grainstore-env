---
title: GrainStore-Env
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# 🌾 GrainStore-Env

An OpenEnv-compatible reinforcement learning environment for smart grain silo monitoring.

An AI agent reads IoT sensor data from a hermetic grain silo and must take correct interventions to prevent grain spoilage and silo damage.

## Tasks

- Task 1 (Easy): Sensor Alert Detection
- Task 2 (Medium): Intervention Selection  
- Task 3 (Hard): Multi-Hazard Response

## Setup
```bash
docker build -t grainstore-env .
docker run -p 7860:7860 grainstore-env
```

## API Endpoints

- GET /health
- POST /reset
- POST /step
- GET /state
- GET /tasks