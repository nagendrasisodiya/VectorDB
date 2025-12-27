import chromadb
from chromadb.utils import embedding_functions

embedding_fun=embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="All-MiniLM-l6-v2"
)

client=chromadb.Client()

collection_name="employee_collection"

# Each dictionary represents an individual employee with comprehensive information
employees_data= [
    {
        "id": "employee_1",
        "name": "John Doe",
        "experience": 5,
        "department": "Engineering",
        "role": "Software Engineer",
        "skills": "Python, JavaScript, React, Node.js, databases",
        "location": "New York",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_2",
        "name": "Jane Smith",
        "experience": 8,
        "department": "Marketing",
        "role": "Marketing Manager",
        "skills": "Digital marketing, SEO, content strategy, analytics, social media",
        "location": "Los Angeles",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_3",
        "name": "Alice Johnson",
        "experience": 3,
        "department": "HR",
        "role": "HR Coordinator",
        "skills": "Recruitment, employee relations, HR policies, training programs",
        "location": "Chicago",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_4",
        "name": "Michael Brown",
        "experience": 12,
        "department": "Engineering",
        "role": "Senior Software Engineer",
        "skills": "Java, Spring Boot, microservices, cloud architecture, DevOps",
        "location": "San Francisco",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_5",
        "name": "Emily Wilson",
        "experience": 2,
        "department": "Marketing",
        "role": "Marketing Assistant",
        "skills": "Content creation, email marketing, market research, social media management",
        "location": "Austin",
        "employment_type": "Part-time"
    },
    {
        "id": "employee_6",
        "name": "David Lee",
        "experience": 15,
        "department": "Engineering",
        "role": "Engineering Manager",
        "skills": "Team leadership, project management, software architecture, mentoring",
        "location": "Seattle",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_7",
        "name": "Sarah Clark",
        "experience": 8,
        "department": "HR",
        "role": "HR Manager",
        "skills": "Performance management, compensation planning, policy development, conflict resolution",
        "location": "Boston",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_8",
        "name": "Chris Evans",
        "experience": 20,
        "department": "Engineering",
        "role": "Senior Architect",
        "skills": "System design, distributed systems, cloud platforms, technical strategy",
        "location": "New York",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_9",
        "name": "Jessica Taylor",
        "experience": 4,
        "department": "Marketing",
        "role": "Marketing Specialist",
        "skills": "Brand management, advertising campaigns, customer analytics, creative strategy",
        "location": "Miami",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_10",
        "name": "Alex Rodriguez",
        "experience": 18,
        "department": "Engineering",
        "role": "Lead Software Engineer",
        "skills": "Full-stack development, React, Python, machine learning, data science",
        "location": "Denver",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_11",
        "name": "Hannah White",
        "experience": 6,
        "department": "HR",
        "role": "HR Business Partner",
        "skills": "Strategic HR, organizational development, change management, employee engagement",
        "location": "Portland",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_12",
        "name": "Kevin Martinez",
        "experience": 10,
        "department": "Engineering",
        "role": "DevOps Engineer",
        "skills": "Docker, Kubernetes, AWS, CI/CD pipelines, infrastructure automation",
        "location": "Phoenix",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_13",
        "name": "Rachel Brown",
        "experience": 7,
        "department": "Marketing",
        "role": "Marketing Director",
        "skills": "Strategic marketing, team leadership, budget management, campaign optimization",
        "location": "Atlanta",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_14",
        "name": "Matthew Garcia",
        "experience": 3,
        "department": "Engineering",
        "role": "Junior Software Engineer",
        "skills": "JavaScript, HTML/CSS, basic backend development, learning frameworks",
        "location": "Dallas",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_15",
        "name": "Olivia Moore",
        "experience": 12,
        "department": "Engineering",
        "role": "Principal Engineer",
        "skills": "Technical leadership, system architecture, performance optimization, mentoring",
        "location": "San Francisco",
        "employment_type": "Full-time"
    },
]

# creating a meaningful document from the above employee data
employee_document=[]
for employee in employees_data:
    document=f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}."
    document +=f"skills: {employee['skills']}. Located in {employee['location']}."
    document +=f"employee type:{employee['employment_type']}"
    employee_document.append(document)

def main():
    global collection
    try:
        collection=client.create_collection(
            name=collection_name,
            metadata={"description":"collection of employee data"},
            configuration={
                "hnsw":{"space":"cosine"},
                "embedding_function":embedding_fun
            }
        )
        collection.add(
            documents=employee_document,
            metadatas=[
                {
                    "name":employee["name"],
                    "department":employee["department"],
                    "role": employee["role"],
                    "experience": employee["experience"],
                    "location": employee["location"],
                    "employment_type": employee["employment_type"]
                }for employee in employees_data
            ],
            ids=[employee["id"] for employee in employees_data]
        )
        pass
    except Exception as error:
        print(f"Exception :{error}")
    print(f"collection created: {collection.name}")
    # getting all the content of collection
    all_items=collection.get()
    print("collection contents :")
    print(f"number of documents: {len(all_items['documents'])}")
    print(f"all the documents in collection{ all_items['documents']}")
    perform_advance_search(collection)


# similarity search implementation
def perform_advance_search(collection):
    try:
        print("------Similarity Search------")
        print("Searching for python developers")
        query_text="Python developer with web development experience"
        results=collection.query(
            query_texts=[query_text],
            n_results=2
        )
        print(f" Query: {query_text}")
        for i, (doc_id, document, distance) in enumerate(zip(
                results['ids'][0], results['documents'][0], results['distances'][0]
        )):
            metadata = results['metadatas'][0][i]
            print(f"  {i + 1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     Role: {metadata['role']}, Department: {metadata['department']}")
            print(f"     Document: {document[:200]}...")

        print("--metadat filtering")
        results=collection.get(
            where={"department":"Engineering"},
        )
        print(results)
        results = collection.get(
            where={"experience": {"$gte": 10}}
        )
        print(results)

        print("Combined Search: Similarity + Metadata Filtering")
        query_text="senior python developer full-stack"
        results=collection.query(
            query_texts=[query_text],
            n_results=5,
            where={
                "$and":[
                    {"experience":{"$gte":8}},
                    {"location":{"$in":["San Francisco", "New York", "Seattle"]}}
                ]
            }
        )
        print(f" query text:{query_text}")
        print(results)
        pass
    except Exception as error:
        print(f"Exception in advance_search: {error}")



if __name__=='__main__':
    main()