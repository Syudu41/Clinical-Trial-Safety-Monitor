"""
Use Existing Community ML Lambda Layer
Instead of building our own, use a pre-built layer with scikit-learn, pandas, numpy
"""

# Known working ML layers for Lambda (US-East-1)
# These are community-maintained layers with scikit-learn, pandas, numpy

AVAILABLE_ML_LAYERS = {
    'us-east-1': [
        # AWSDataWrangler layer (includes pandas, numpy)
        'arn:aws:lambda:us-east-1:336392948345:layer:AWSDataWrangler-Python39:12',
        
        # Scientific computing layer (pandas, numpy, scipy, scikit-learn)
        'arn:aws:lambda:us-east-1:113088814899:layer:Klayers-p39-scikit-learn:1',
        
        # Pandas + NumPy layer
        'arn:aws:lambda:us-east-1:336392948345:layer:AWSDataWrangler-Python311:1'
    ],
    'us-west-2': [
        'arn:aws:lambda:us-west-2:336392948345:layer:AWSDataWrangler-Python39:12'
    ]
}

def get_ml_layer_arn(region='us-east-1'):
    """Get a working ML layer ARN for your region"""
    layers = AVAILABLE_ML_LAYERS.get(region, AVAILABLE_ML_LAYERS['us-east-1'])
    
    print("🎯 Available ML Layers:")
    for i, layer in enumerate(layers, 1):
        print(f"   {i}. {layer}")
    
    # Recommend the first one (usually most stable)
    recommended = layers[0]
    print(f"\n✅ Recommended Layer: {recommended}")
    
    return recommended

if __name__ == "__main__":
    print("Pre-Built ML Lambda Layers")
    print("=" * 40)
    
    layer_arn = get_ml_layer_arn()
    
    print(f"\n🚀 Use this Layer ARN for deployment:")
    print(f"📋 {layer_arn}")
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Copy the layer ARN above")
    print(f"2. Run: python scripts/deploy_lambda_with_layers.py \"{layer_arn}\"")
    print(f"3. Skip the layer creation entirely!")