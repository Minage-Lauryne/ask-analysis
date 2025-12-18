# complere-agent
complere agent backend

## Prerequisites

- Python 3.9.5 or higher
- pip (Python package manager)
- Git

## Installation & Setup

## Clone the Repository

```bash
git clone https://github.com/complere-llc/complere-agent.git
cd complere-agent

# Create virtual environment
python3 -m venv venv

# Activate virtual environment:

# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate

## Install requirements
pip install -r requirements.txt

# environment file
Find the .env file on the engineering channel or ask either Hunter, Saloi, Yordanos or Minage to share


### Run the application
# Fastapi server
uvicorn app.main:app --reload --port 8080
