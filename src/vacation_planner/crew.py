from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
import os


app = BedrockAgentCoreApp()

# Initialize SerperDevTool
serper_dev_tool = SerperDevTool(api_key=os.environ.get("SERPER_DEV_API_KEY", ""))

llm = LLM(model=os.environ.get("MODEL", "gpt-4"))


@CrewBase
class VacationPlanner():
    """VacationPlanner crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def vacation_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['vacation_researcher'], # type: ignore[index]
            verbose=True,
            tools=[serper_dev_tool],
            llm=llm
        )

    @agent
    def itinerary_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['itinerary_planner'], # type: ignore[index]
            verbose=True,
            llm=llm
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the VacationPlanner crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

@app.entrypoint # bedrock_agentcore decorator to define the entrypoint for the agent
def agent_invocation(payload, context):
    """Handler for agents invocation"""
    print(f"Payload: {payload}")

    try:
        # Extract user input from the payload
        user_input = payload.get('user_input', '')
        if not user_input:
            return {"error": "No user input provided."}
        print(f"Processing vacation destination: {user_input}")

        # Crew execution - Create an instance of the VacationPlanner crew and run crew method
        research_crew_instance = VacationPlanner()
        crew = research_crew_instance.crew()

        # Start the sequential agent workflow
        result = crew.kickoff(inputs={"topic": user_input})

        print("Context:\n-----------\n", context)
        print("Result raw:\n***********\n", result.raw)

        # Safely access json_dict if exists
        if hasattr(result, "json_dict"):
            print("Result JSON:\n***********\n", result.json_dict)

            return {"result": result.raw}

    except Exception as e:
        print(f"Error during crew execution: {e}")
        return {"error": f"Error during crew execution: {str(e)}"}


if __name__ == "__main__":
    # Run AgentCore server - HTTP server on port 8080
    app.run()