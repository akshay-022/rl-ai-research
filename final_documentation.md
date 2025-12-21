## Final Documentation

I want to create RL environments for an automated AI researcher. I find this very interesting. 

For this take home I want to focus on those skills which are the most widely generalisable for an AI researcher. I do not want to dive into implementations of narrow techniques (like optimizing a cuda kernel). Those would be lower order bits to work on. 

I want to create the foundations for the following 3 things : 
1. How well an LLM can do a literature survey and understand the full scope of a field. 
2. How good is it's research taste - i.e. when it proposes a direction to research in, how well is it rooted in results of current literature (proxy for likelihood of success of proposed approach), and how impactful the proposed experiment would be to the field if successful (proxy for importance of proposed approach).
3. Implementation of SOTA evals proposed in other papers.

The above 3 are fairly universal skills for any very good researcher, irrespective of methodology or field. 

Task 1 - LLM literature survey : 

Tools : 
1. Give it access to tools to search research papers onine.
2. Give it the abililty to download any paper and read it. 

Evaluation : 
I want to evaluate along a few dimensions - 
1. How comprehensively it has covered papers in the different areas of that field. 
2. How well it has encapsulated the efficacy of all these different approaches and where the boundaries of those approaches are. 
3. That it has cited all the seminal papers in the space. 
4. Additional points if it has looked at adjacent fields and drawn parallels. Like Ilya is trying to do by comparing AI to the human brain. 
5. Potential reward if it analyses the promise and business viability in each of these directions of research.

How do we do this evaluation?
1. The best way seems to be to use the best researchers to create a SoTA dataset for this. I am using a proxy - using deep research for a few different topics and assigning a reward signal as percentage of distinct spaces of the field the llm dove into. 
2. LLM as a judge to see if it included the results of the most important papers and pros and cons of approaches in the real world. 

Task 2 : 

Tools : 
1. The same tools it had access to during the LLM literature survey. 

Evaluation : 
1. Do a comprehensive literature survey with the LLM first. Then see how efficient it's proposed approach has been traditionally. 
2. Use llm as a judge to understand how efficient the proposed approach could be. If extrapolated 5 years into the future where could this research go. 
3. LLM as a judge to analyse what the business viability of the idea is.


Task 3 : 

Tools : 
1. Tool to get the repository for an eval from github.
2. Tool to write to files. 

Evaluation : 
1. This is straightforward. 
    a. Does the eval run. 
    b. Is it implemented correctly. 






