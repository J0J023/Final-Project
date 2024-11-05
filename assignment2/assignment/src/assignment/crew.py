from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
import os
os.environ['SERPER_API_KEY']='8379972250883034c50c38bb28455fa2b1166d56'

search_tool=SerperDevTool()

@CrewBase
class AssignmentCrew():
	"""Assignment crew"""

	@agent
	def search_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['search_agent'],
			tools=[search_tool],
			verbose=True
		)

	@agent
	def empathiser (self) -> Agent:
		return Agent(
			config=self.agents_config['empathiser'],
			verbose=True
		)
	
	@agent
	def insight_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['insight_agent'],
			verbose=True
		)
	
	# @agent
	# def cognitive_reframer(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['cognitive_reframer'],
	# 		verbose=True
	# 	)
	
	@agent
	def mindfulness_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['mindfulness_agent'],
			verbose=True
		)

	@task
	def web_search(self) -> Task:
		return Task(
			config=self.tasks_config['web_search'],
		)

	@task
	def empathising(self) -> Task:
		return Task(
			config=self.tasks_config['empathising'],
			output_file='diagnosis.md'
		)
	
	@task
	def providing_insight(self) -> Task:
		return Task(
			config=self.tasks_config['providing_insight'],
			output_file='diagnosis.md'
		)
		
	@task
	def mindfulness_techniques(self) -> Task:
		return Task(
			config=self.tasks_config['mindfulness_techniques'],
			output_file='diagnosis.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Assignment crew"""
		return Crew(
			agents=self.agents, 
			tasks=self.tasks, 
			process=Process.sequential,
			verbose=True,
		)