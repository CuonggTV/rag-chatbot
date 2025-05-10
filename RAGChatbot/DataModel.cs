using Microsoft.Extensions.VectorData;

namespace ClaimRequest.AI;

public class DataModel
{
    [VectorStoreRecordKey]
    public required string Key { get; init; }
    
    [VectorStoreRecordData]
    public required string Title { get; init; }
    
    [VectorStoreRecordData(IsFullTextIndexed = true)]
    public required string Text { get; init; }
    
    [VectorStoreRecordVector(3072)]
    public ReadOnlyMemory<float> TextEmbedding { get; init; }
}