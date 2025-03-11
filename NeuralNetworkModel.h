

// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"

#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNEModelData.h"
#include "Engine/TextureRenderTarget2D.h"

#include "NeuralNetworkModel.generated.h"


// Create the model from a neural network model data asset
//TObjectPtr<UNNEModelData> ModelData = LoadObject<UNNEModelData>(GetTransientPackage(), TEXT("/Users/jordanneff/Desktop/PROJECT/src/YOLO/yolov8n.onnx"));
//TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(FString("NNERuntimeORTCpu"));
//TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance = Runtime->CreateModelCPU(ModelData)->CreateModelInstanceCPU();


USTRUCT(BlueprintType, Category = "NNE - Tutorial")
struct FNeuralNetworkTensor
{
    GENERATED_BODY()

public:

    UPROPERTY(BlueprintReadWrite, Category = "NNE - Tutorial")
    TArray<int32> Shape = TArray<int32>();

    UPROPERTY(BlueprintReadWrite, Category = "NNE - Tutorial")
    TArray<float> Data = TArray<float>();
};

UCLASS(BlueprintType, Category = "NNE - Tutorial")
class TUTORIAL_API UNeuralNetworkModel : public UObject
{
    GENERATED_BODY()

public:

    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    static TArray<FString> GetRuntimeNames();

    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    static UNeuralNetworkModel* CreateModel(UObject* Parent, FString RuntimeName, UNNEModelData* ModelData);

    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    static bool CreateTensor(TArray<int32> Shape, UPARAM(ref) FNeuralNetworkTensor& Tensor);

public:

    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    int32 NumInputs();

    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    int32 NumOutputs();

    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    TArray<int32> GetInputShape(int32 Index);

    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    TArray<int32> GetOutputShape(int32 Index);

public:

    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    bool SetInputs(const TArray<FNeuralNetworkTensor>& Inputs);

    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    bool RunSync(UPARAM(ref) TArray<FNeuralNetworkTensor>& Outputs);
    
    //UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    //bool ConvertPngToTensorInput(const FString& PngFilePath);
    
    //UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    //static TArray<float> ProcessRenderTarget(UTextureRenderTarget2D* RenderTarget);
    
    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    TArray<uint8> ProcessScreenshot();
    
    UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
    TArray<uint8> LoadPNGToPixelArray(const FString& FilePath, int32& OutWidth, int32& OutHeight);


private:
    TSharedPtr<UE::NNE::IModelCPU> Model;
    TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;
    TArray<UE::NNE::FTensorBindingCPU> InputBindings;
    TArray<UE::NNE::FTensorShape> InputShapes;

};



// Prepare the model given a certain input size
//ModelInstance->SetInputTensorShapes(InputShapes);
 
// Run the model passing caller owned CPU memory
//ModelInstance->RunSync(Inputs, Outputs);


