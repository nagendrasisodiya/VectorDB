import os

from shared_functions import *

from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()

food_items=[]

my_credentials = {
    "url": "https://eu-de.ml.cloud.ibm.com",
    "apikey": os.getenv("WATSONX_API_KEY")
}
model = ModelInference(
    model_id='ibm/granite-4-h-small',
    credentials=my_credentials,
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params={'max_new_tokens':400},
    space_id=None,
    verify=False,
)


def main():
    try:
        print("🤖 Enhanced RAG-Powered Food Recommendation Chatbot")
        print("=" * 55)

        global food_items
        food_items=load_food_data(r"C:\Users\ASUS\OneDrive\Desktop\GEN-AI\VectorDB\food search boot\FoodDataSet.json")
        collection=create_similarity_search_collection(
            "enhanced_rag_food_chatbot",
            {'description': 'Enhanced RAG chatbot with IBM watsonx.ai integration'}
        )

        populate_similarity_collection(collection, food_items)
        print("✅ Vector database ready")

        print("testing llm connection...")
        test_response=model.generate("Hello", params=None)
        if test_response and "results" in test_response:
            print("✅✅ llm connection established successfully")
        else:
            print("❌❌ connection failed ")

        # starting conversation
        enhanced_rag_food_chatbot(collection)
    except Exception as error:
        print(f"exception in rag-chat-boot: {error}")


def prepare_context_for_llm(query:str, search_results:List[Dict])->str:
    if not search_results:
        return "no food item found in search result"
    context_parts= []
    context_parts.append(f"based on your query, here are most relevent food item in our database:")
    context_parts.append("")

    for i, result in enumerate(search_results[:3], 1):
        food_context=[]
        food_context.append(f"Option {i}: {result['food_name']}")
        food_context.append(f"  - Description: {result['food_description']}")
        food_context.append(f"  - Cuisine: {result['cuisine_type']}")
        food_context.append(f"  - Calories: {result['food_calories_per_serving']} per serving")

        if result.get('food_ingredients'):
            ingredients = result['food_ingredients']
            if isinstance(ingredients, list):
                food_context.append(f"  - Key ingredients: {', '.join(ingredients[:5])}")
            else:
                food_context.append(f"  - Key ingredients: {ingredients}")

        if result.get('food_health_benefits'):
            food_context.append(f"  - Health benefits: {result['food_health_benefits']}")

        if result.get('cooking_method'):
            food_context.append(f"  - Cooking method: {result['cooking_method']}")

        if result.get('taste_profile'):
            food_context.append(f"  - Taste profile: {result['taste_profile']}")

        food_context.append(f"  - Similarity score: {result['similarity_score'] * 100:.1f}%")
        food_context.append("")

        context_parts.extend(food_context)

    return "\n".join(context_parts)



def generate_llm_rag_response(query:str, search_results:List[Dict])->str:
    try:
        context=prepare_context_for_llm(query, search_results)

        prompt = f'''You are a helpful food recommendation assistant. A user is asking for food recommendations, and I've retrieved relevant options from a food database.
        User Query: "{query}"
        Retrieved Food Information:
        {context}
        Please provide a helpful, short response that:
        1. Acknowledges the user's request
        2. Recommends 2-3 specific food items from the retrieved options
        3. Explains why these recommendations match their request
        4. Includes relevant details like cuisine type, calories, or health benefits
        5. Uses a friendly, conversational tone
        6. Keeps the response concise but informative
        Response:'''

        generated_response=model.generate(
            prompt=prompt,
            params=None
        )

        if generated_response and "results" in generated_response:
            response_text=generated_response["results"][0]["generated_text"]
            response_text=response_text.strip()

            # if response is to shorter
            if len(response_text)<50:
                return generate_fallback_response(query, search_results)
            else:
                return response_text
        else:
            return generate_fallback_response(query, search_results)

    except Exception as error:
        print(f"❌❌ LLM error")
        return generate_fallback_response(query, search_results)


def generate_fallback_response(query, search_results)->str:
    if not search_results:
        return "I couldn't find any food items matching your request. Try describing what you're in the mood for with different words!"
    top_result=search_results[0]
    response_parts=[]
    response_parts.append(f"Based on your request for '{query}', I'd recommend {top_result['food_name']}.")
    response_parts.append(
        f"It's a {top_result['cuisine_type']} dish with {top_result['food_calories_per_serving']} calories per serving.")

    if len(search_results) > 1:
        second_choice = search_results[1]
        response_parts.append(f"Another great option would be {second_choice['food_name']}.")

    return " ".join(response_parts)

def enhanced_rag_food_chatbot(collection):
    """Enhanced RAG-powered conversational food chatbot with IBM Granite"""
    print("\n" + "="*70)
    print("🤖 ENHANCED RAG FOOD RECOMMENDATION CHATBOT")
    print("   Powered by IBM's Granite Model")
    print("="*70)
    print("💬 Ask me about food recommendations using natural language!")
    print("\nExample queries:")
    print("  • 'I want something spicy and healthy for dinner'")
    print("  • 'What Italian dishes do you recommend under 400 calories?'")
    print("  • 'I'm craving comfort food for a cold evening'")
    print("  • 'Suggest some protein-rich breakfast options'")
    print("\nCommands:")
    print("  • 'help' - Show detailed help menu")
    print("  • 'compare' - Compare recommendations for two different queries")
    print("  • 'quit' - Exit the chatbot")
    print("-" * 70)

    conversation_history=[]

    while True:
        try:
            user_input=input("\n🧒 YOU:")

            if not user_input:
                print("🤖 Bot: Please tell me what kind of food you're looking for!")
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🤖 Bot: Thank you for using the Enhanced RAG Food Chatbot!")
                print("      Hope you found some delicious recommendations! 👋")
                break

            elif user_input.lower() in ['help', 'h']:
                show_enhanced_rag_help()
            else:
                # Process the food query with enhanced RAG
                handle_enhanced_rag_query(collection, user_input, conversation_history)
                conversation_history.append(user_input)

            # Keep conversation history manageable
            if len(conversation_history) > 5:
                conversation_history = conversation_history[-3:]

        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! Hope you find something delicious! 👋")
            break
        except Exception as e:
            print(f"❌ Bot: Sorry, I encountered an error: {e}")

def handle_enhanced_rag_query(collection, query: str, conversation_history: List[str]):
    print(f"\n🔍 Searching vector database for: '{query}'...")
    search_results=perform_similarity_search(collection, query, 3)

    if not search_results:
        print("🤖 Bot: I couldn't find any food items matching your request.")
        print("      Try describing what you're in the mood for with different words!")
        return

    print(f"✅ Found {len(search_results)} relevant matches")
    print("🧠 Generating AI-powered response...")
    ai_response=generate_llm_rag_response(query, search_results)
    print(f"\n🤖 Bot: {ai_response}")


    # Show detailed results for reference
    print(f"\n📊 Search Results Details:")
    print("-" * 45)
    for i, result in enumerate(search_results[:3], 1):
        print(f"{i}. 🍽️  {result['food_name']}")
        print(f"   📍 {result['cuisine_type']} | 🔥 {result['food_calories_per_serving']} cal | 📈 {result['similarity_score']*100:.1f}% match")
        if i < 3:
            print()

def show_enhanced_rag_help():
        """Display help information for enhanced RAG chatbot"""
        print("\n📖 ENHANCED RAG CHATBOT HELP")
        print("=" * 45)
        print("🧠 This chatbot uses IBM Granite to understand your")
        print("   food preferences and provide intelligent recommendations.")
        print("\nHow to get the best recommendations:")
        print("  • Be specific: 'healthy Italian pasta under 350 calories'")
        print("  • Mention preferences: 'spicy comfort food for cold weather'")
        print("  • Include context: 'light breakfast for busy morning'")
        print("  • Ask about benefits: 'protein-rich foods for workout recovery'")
        print("\nSpecial features:")
        print("  • 🔍 Vector similarity search finds relevant foods")
        print("  • 🧠 AI analysis provides contextual explanations")
        print("  • 📊 Detailed nutritional and cuisine information")
        print("  • 🔄 Smart comparison between different preferences")
        print("\nCommands:")
        print("  • 'compare' - AI-powered comparison of two queries")
        print("  • 'help' - Show this help menu")
        print("  • 'quit' - Exit the chatbot")
        print("\nTips for better results:")
        print("  • Use natural language - talk like you would to a friend")
        print("  • Mention dietary restrictions or preferences")
        print("  • Include meal timing (breakfast, lunch, dinner)")
        print("  • Specify if you want healthy, comfort, or indulgent options")

if __name__ == "__main__":
        main()
