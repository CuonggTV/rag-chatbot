namespace ClaimRequest.AI;

public interface IRAGChatService
{
    public  Task<string> Answer(UserArugments userArugments, string question);
}