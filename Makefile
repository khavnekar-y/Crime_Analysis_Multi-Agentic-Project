.PHONY: install prerequisites backend frontend rag docker-up docker-down
# Install project dependencies using Poetry 
install:
	poetry install

lock:
	poetry lock
update:
	poetry update



snowflake:
	poetry run python .\agents\snowflake_agent.py
	
backend:
	poetry run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
frontend:
	poetry run streamlit run streamlit_app.py --server.port 8501
pinecone:
	poetry run python .\Rag_modelings\rag_pinecone.py
pinecone-test:
	poetry run python .\Rag_modelings\Chunking_Stats\pinecone\pinecone_.py


llm:
	poetry run python .\Backend\litellm_query_generator.py

ragagent:
	@echo "Running RAG Agent..."
	poetry run python .\agents\rag_agent.py

# Build commands
fastapibuild:
	docker build --platform=linux/amd64 -t gcr.io/gen-lang-client-0567410120/fastapi-agentic-app .

streamlitbuild:
	docker build --platform=linux/amd64 -t gcr.io/gen-lang-client-0567410120/streamlit-agentic-app .

# Run commands
fastapirun:
	docker run --name fastapi-app -p 8080:8080 gcr.io/gen-lang-client-0567410120/fastapi-agentic-app

streamlitrun:
    docker run --name streamlit-app -p 8501:8501 gcr.io/gen-lang-client-0567410120/streamlit-agentic-app

# Stop commands
fastapistop:
	docker stop fastapi-app

streamlitstop:
	docker stop streamlit-app

# Remove containers
fastapiremove:
	docker rm fastapi-app

streamlitremove:
	docker rm streamlit-app

# Remove images
fastapiimage-remove:
	docker rmi gcr.io/gen-lang-client-0567410120/fastapi-agentic-app

streamlitimage-remove:
	docker rmi gcr.io/gen-lang-client-0567410120/streamlit-agentic-app

# Push images to GCR
fastapipush:
	docker push gcr.io/gen-lang-client-0567410120/fastapi-agentic-app

streamlitpush:
	docker push gcr.io/gen-lang-client-0567410120/streamlit-agentic-app

# Pull images from GCR
fastapipull:
	docker pull gcr.io/gen-lang-client-0567410120/fastapi-agentic-app

streamlitpull:
	docker pull gcr.io/gen-lang-client-0567410120/streamlit-agentic-app

# Combined commands
cleanup: 
	fastapistop streamlitstop fastapiremove streamlitremove

rebuild: 
	cleanup fastapibuild streamlitbuild

fastapideploy:
	gcloud run deploy fastapi-agentic-app --image gcr.io/gen-lang-client-0567410120/fastapi-agentic-app --platform managed --region us-central1 --allow-unauthenticated

streamlitdeploy:
	gcloud run deploy streamlit-agentic-app --image gcr.io/gen-lang-client-0567410120/streamlit-agentic-app --platform managed --region us-central1 --allow-unauthenticated


# SerAPI Web API
websearch:
	@echo "Running NVIDIA Web Search Agent..."
	poetry run python agents/websearch_agent.py

webagent-news:
	@echo "Running NVIDIA News Search..."
	poetry run python -c "from agents.websearch_agent import NvidiaWebSearchAgent; import json; agent = NvidiaWebSearchAgent(); results = agent.search_nvidia_news(days_ago=7); print(json.dumps(results, indent=2))"

webagent-financial:
	@echo "Running NVIDIA Financial Search..."
	poetry run python -c "from agents.websearch_agent import NvidiaWebSearchAgent; import json; agent = NvidiaWebSearchAgent(); results = agent.search_nvidia_financial(); print(json.dumps(results, indent=2))"

webagent-quarterly:
	@echo "Running NVIDIA Quarterly Report Search..."
	poetry run python -c "from agents.websearch_agent import NvidiaWebSearchAgent; import json; agent = NvidiaWebSearchAgent(); results = agent.search_quarterly_report_info(year=2023, quarter=4); print(json.dumps(results, indent=2))"

webagent-general:
	@echo "Running NVIDIA General Information Search..."
	poetry run python -c "from agents.websearch_agent import NvidiaWebSearchAgent; import json; agent = NvidiaWebSearchAgent(); results = agent.search_nvidia_general(); print(json.dumps(results, indent=2))"

webagent-report:
	@echo "Generating comprehensive NVIDIA research report..."
	poetry run python -c "from agents.websearch_agent import NvidiaWebSearchAgent; import datetime; agent = NvidiaWebSearchAgent(); current_year = datetime.datetime.now().year; current_quarter = (datetime.datetime.now().month - 1) // 3 + 1; report = agent.generate_research_report(year=current_year, quarter=current_quarter); print(f'Report saved to: {report.get(\"saved_to\", \"unknown path\")}');"

pipeline:
	@echo "Running Language Graph Agent..."
	poetry run python .\langGraph\pipeline.py