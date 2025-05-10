using ClaimRequest.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;

#pragma warning disable SKEXP0070

// API Key and Model Name
string GeminiAPIKey = "";
var modelName = "gemini-2.0-flash";
var embeddingModelName = "gemini-embedding-exp-03-07";

// MongoDB Connection String
var MongoDBConnectionString = "";
var MongoDBDatabaseName = "claim-request";

var kernelBuilder = Kernel.CreateBuilder();

kernelBuilder.Services.AddGoogleAIGeminiChatCompletion(modelName, GeminiAPIKey);
kernelBuilder.Services.AddGoogleAIEmbeddingGeneration(embeddingModelName, GeminiAPIKey);
kernelBuilder.Services.AddMongoDBVectorStore(MongoDBConnectionString, MongoDBDatabaseName);


// Build the kernel
var kernel = kernelBuilder.Build();

kernelBuilder.Services.AddSingleton(kernel);
kernelBuilder.Services.AddSingleton<IRAGChatService, RAGChatService>();
var serviceProvider = kernelBuilder.Services.BuildServiceProvider();

var ragChatService = serviceProvider.GetRequiredService<IRAGChatService>();

// Begin Chat
while (true)
{
    Console.Write("Question: ");
    var question = Console.ReadLine()!;

    var answer = await ragChatService.Answer(
        new UserArugments
        {
            Email = "newstaff@gmail.com",
            Name = "new staff",
            Role = "staff "
        }, question);
    Console.WriteLine(answer);
}

