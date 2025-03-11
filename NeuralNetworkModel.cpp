

// Fill out your copyright notice in the Description page of Project Settings.


#include "NeuralNetworkModel.h"

///*
#include "IImageWrapperModule.h"
#include "IImageWrapper.h"
#include "Modules/ModuleManager.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/FileHelper.h"

#include "Engine/TextureRenderTarget2D.h"
#include "Engine/Texture2D.h"
//#include "Rendering/TextureRenderTargetResource.h"
#include "Kismet/KismetRenderingLibrary.h"

//*/

// Fixed Input Shape for YOLOv8
const TArray<int32> FIXED_INPUT_SHAPE = {1, 3, 640, 640};

// ######################################################################################################################

/*
- Parameters: None.
- What it does: Fetches and logs the names of available ONNX runtime backends (e.g., CPU, DML).
- Return Value: TArray<FString> containing the names of available runtimes.
*/
TArray<FString> UNeuralNetworkModel::GetRuntimeNames()
{
    UE_LOG(LogTemp, Log, TEXT("Fetching available runtime names"));

    TArray<FString> RuntimeNames = UE::NNE::GetAllRuntimeNames();

    UE_LOG(LogTemp, Warning, TEXT("Number of runtimes found: %d"), RuntimeNames.Num());
    // Log the names of the available runtimes
    for (const FString& RuntimeName : RuntimeNames)
    {
        UE_LOG(LogTemp, Warning, TEXT("Available runtime: %s"), *RuntimeName);
    }

    return RuntimeNames;
}

// ######################################################################################################################
// ######################################################################################################################

/*
- Parameters:
 1) Parent: The parent UObject for the model instance.
 2) RuntimeName: The name of the runtime to use (e.g., "NNERuntimeORTCpu").
 3) ModelData: The ONNX model data loaded as a UNNEModelData asset.
- What it does: Creates and initializes a neural network model instance using the specified runtime and model data.
- Return Value: UNeuralNetworkModel* pointing to the created model instance, or nullptr if creation fails.
 */
UNeuralNetworkModel* UNeuralNetworkModel::CreateModel(UObject* Parent, FString RuntimeName, UNNEModelData* ModelData)
{
    using namespace UE::NNE;

    // Explicitly set the desired runtime
    RuntimeName = "NNERuntimeORTCpu";
    
    // Load ModelData from Unreal's asset system
    ModelData = LoadObject<UNNEModelData>(nullptr, TEXT("/Game/yolov8n"));
    if (!ModelData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to load ONNX model data."));
        return nullptr;
    }

    TWeakInterfacePtr<INNERuntimeCPU> Runtime = GetRuntime<INNERuntimeCPU>(RuntimeName);
    if (!Runtime.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("No CPU runtime '%s' found"), *RuntimeName);
        return nullptr;
    }
    UE_LOG(LogTemp, Warning, TEXT("Creating model using runtime: %s"), *RuntimeName);

    TSharedPtr<UE::NNE::IModelCPU> UniqueModel = Runtime->CreateModelCPU(ModelData);
    if (!UniqueModel.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Could not create the CPU model"));
        return nullptr;
    }
    UE_LOG(LogTemp, Warning, TEXT("Created model using runtime: %s"), *RuntimeName);

    // Create a new instance of UNeuralNetworkModel
    UNeuralNetworkModel* Result = NewObject<UNeuralNetworkModel>(Parent);
    if (!Result)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create UNeuralNetworkModel instance."));
        return nullptr;
    }

    UE_LOG(LogTemp, Warning, TEXT("Successfully created UNeuralNetworkModel instance."));
    
    if (UniqueModel.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("Model is valid after creation."));
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Model is NOT valid after creation."));
    }

    // Store the model in the created instance
    Result->Model = UniqueModel;

    // Store the ModelInstance by creating a new instance from the model
    Result->ModelInstance = UniqueModel->CreateModelInstanceCPU();
    if (!Result->ModelInstance.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create ModelInstance."));
        return nullptr;
    }

    UE_LOG(LogTemp, Warning, TEXT("ModelInstance successfully created and stored in UNeuralNetworkModel."));

    return Result;
}



// ######################################################################################################################

/*
- Parameters:
 1) Shape: The shape of the tensor as a TArray<int32>.
 2) Tensor: A reference to a FNeuralNetworkTensor to be populated.
- What it does: Allocates memory for a tensor with the specified shape and initializes its data array.
- Return Value: bool indicating whether the tensor was successfully created.
 */
bool UNeuralNetworkModel::CreateTensor(TArray<int32> Shape, UPARAM(ref) FNeuralNetworkTensor& Tensor)
{
    UE_LOG(LogTemp, Warning, TEXT("CreateTensor called with shape: %s"), *FString::JoinBy(Shape, TEXT(", "), [](int32 Val) { return FString::FromInt(Val); }));

    if (Shape.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("CreateTensor failed: Shape array is empty"));
        return false;
    }

    int32 Volume = 1;
    for (int32 i = 0; i < Shape.Num(); i++)
    {
        if (Shape[i] < 1)
        {
            UE_LOG(LogTemp, Error, TEXT("CreateTensor failed: Invalid shape value at index %d (value: %d)"), i, Shape[i]);
            return false;
        }
        Volume *= Shape[i];
    }

    UE_LOG(LogTemp, Warning, TEXT("CreateTensor: Allocating tensor with volume %d"), Volume);

    Tensor.Shape = Shape;
    Tensor.Data.SetNum(Volume);

    UE_LOG(LogTemp, Warning, TEXT("CreateTensor: Successfully created tensor."));
    return true;
}

/*
 - Parameters: None.
 - What it does: Returns the number of input tensors expected by the model.
 - Return Value: int32 representing the number of input tensors.
 */
int32 UNeuralNetworkModel::NumInputs()
{
    check(Model.IsValid());
    int32 Num = ModelInstance->GetInputTensorDescs().Num();
    UE_LOG(LogTemp, Warning, TEXT("NumInputs called: %d inputs found"), Num);
    return Num;
}

/*
 - Parameters: None.
 - What it does: Returns the number of output tensors produced by the model.
 - Return Value: int32 representing the number of output tensors.
 */
int32 UNeuralNetworkModel::NumOutputs()
{
    check(Model.IsValid());
    int32 Num = ModelInstance->GetOutputTensorDescs().Num();
    UE_LOG(LogTemp, Warning, TEXT("NumOutputs called: %d outputs found"), Num);
    return Num;
}

/*
 - Parameters:
    1) Index: The index of the input tensor.
 - What it does: Returns the fixed input shape (1, 3, 640, 640) required by the YOLOv8 model.
 - Return Value: TArray<int32> representing the input tensor shape.
 */
TArray<int32> UNeuralNetworkModel::GetInputShape(int32 Index)
{
    
    UE_LOG(LogTemp, Warning, TEXT("GetInputShape called, returning fixed shape."));
    return FIXED_INPUT_SHAPE;


    /*
     using namespace UE::NNE;

    TConstArrayView<FTensorDesc> Desc = ModelInstance->GetInputTensorDescs();
    if (Index < 0 || Index >= Desc.Num())
    {
        UE_LOG(LogTemp, Error, TEXT("GetInputShape failed: Index %d out of bounds"), Index);
        return TArray<int32>();
    }

    TArray<int32> Shape = TArray<int32>(Desc[Index].GetShape().GetData());
    UE_LOG(LogTemp, Warning, TEXT("GetInputShape called for index %d: Shape = %s"), Index, *FString::JoinBy(Shape, TEXT(", "), [](int32 Val) { return FString::FromInt(Val); }));

    return Shape;
    */
}

////////////////

/*
 - Parameters:
    1) Index: The index of the output tensor.
 - What it does: Returns the fixed output shape (1, 84, 5) expected from the YOLOv8 model.
 - Return Value: TArray<int32> representing the output tensor shape.
 */
TArray<int32> UNeuralNetworkModel::GetOutputShape(int32 Index)
{
    UE_LOG(LogTemp, Warning, TEXT("GetOutputShape called. Using fixed YOLOv8 output shape."));

    // YOLOv8 output shape: (1, 84, N) where N depends on the number of detections.
    return {1, 84, 5}; // Adjust '5' as needed based on your model # of trained objects.
}

/*
TArray<int32> UNeuralNetworkModel::GetOutputShape(int32 Index)
{
    check(Model.IsValid());

    using namespace UE::NNE;

    TConstArrayView<FTensorDesc> Desc = ModelInstance->GetOutputTensorDescs();
    if (Index < 0 || Index >= Desc.Num())
    {
        UE_LOG(LogTemp, Error, TEXT("GetOutputShape failed: Index %d out of bounds"), Index);
        return TArray<int32>();
    }

    TArray<int32> Shape = TArray<int32>(Desc[Index].GetShape().GetData());
    UE_LOG(LogTemp, Warning, TEXT("GetOutputShape called for index %d: Shape = %s"), Index, *FString::JoinBy(Shape, TEXT(", "), [](int32 Val) { return FString::FromInt(Val); }));

    return Shape;
}
 */




/*
 - Parameters:
    1) Inputs: An array of FNeuralNetworkTensor containing the input tensors.
 - What it does: Assigns the input tensors to the model and sets their shapes for inference.
 - Return Value: bool indicating whether the inputs were successfully set.
 */
bool UNeuralNetworkModel::SetInputs(const TArray<FNeuralNetworkTensor>& Inputs)
{
    UE_LOG(LogTemp, Warning, TEXT("SetInputs called"));
    check(Model.IsValid());

    UE_LOG(LogTemp, Warning, TEXT("SetInputs called with %d input tensors"), Inputs.Num());

    using namespace UE::NNE;

    InputBindings.Reset();
    InputShapes.Reset();
    
    // Define the fixed input shape (YOLOv8 expects 1x3x640x640)
    const TArray<int32> FixedShape = {1, 3, 640, 640};
    const int32 ExpectedNumElements = 1 * 3 * 640 * 640;  // 1,228,800 elements


    TConstArrayView<FTensorDesc> InputDescs = ModelInstance->GetInputTensorDescs();
    if (InputDescs.Num() != Inputs.Num())
    {
        UE_LOG(LogTemp, Error, TEXT("SetInputs failed: Expected %d input tensors, but got %d"), InputDescs.Num(), Inputs.Num());
        return false;
    }

    InputBindings.SetNum(Inputs.Num());
    InputShapes.SetNum(Inputs.Num());

    for (int32 i = 0; i < Inputs.Num(); i++)
    {
        
        
        // how do i populate the input tensor with the pixel data i got in ProcessRenderTarget
        
        // Directly assign the input data, assuming it is already in the correct shape
        InputBindings[i].Data = (void*)Inputs[i].Data.GetData();
        InputBindings[i].SizeInBytes = Inputs[i].Data.Num() * sizeof(float);
        
        
        
        
        // tests for image loading
        UE_LOG(LogTemp, Warning, TEXT("SetInputs: InputBindings[i].SizeInBytes %d"),Inputs[i].Data.Num());
        UE_LOG(LogTemp, Warning, TEXT("SetInputs: (void*)Inputs[i].Data.GetData() %d"), (void*)Inputs[i].Data.GetData());


        // Assign the fixed shape to the tensor
        InputShapes[i] = FTensorShape::MakeFromSymbolic(FSymbolicTensorShape::Make(FixedShape));

        UE_LOG(LogTemp, Warning, TEXT("SetInputs: Tensor %d -> Shape: %s, Size: %d bytes"),
            i,
            *FString::JoinBy(FixedShape, TEXT(", "), [](int32 Val) { return FString::FromInt(Val); }),
            InputBindings[i].SizeInBytes);
    }

    if (ModelInstance->SetInputTensorShapes(InputShapes) != UE::NNE::EResultStatus::Ok)
    {
        UE_LOG(LogTemp, Error, TEXT("SetInputs failed: Could not set input tensor shapes"));
        return false;
    }

    UE_LOG(LogTemp, Warning, TEXT("SetInputs: Successfully set input tensors with fixed shape."));
    return true;
}

/*
 - Parameters:
    1) Outputs: A reference to an array of FNeuralNetworkTensor to store the model's output.
 - What it does: Runs the model synchronously using the provided inputs and populates the output tensors.
 - Return Value: bool indicating whether the inference was successful.
 */
bool UNeuralNetworkModel::RunSync(UPARAM(ref) TArray<FNeuralNetworkTensor>& Outputs)
{
    UE_LOG(LogTemp, Warning, TEXT("RunSync called"));

    if (!Model.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("RunSync failed: Model instance is invalid!"));
        return false;
    }

    using namespace UE::NNE;

    TConstArrayView<FTensorDesc> OutputDescs = ModelInstance->GetOutputTensorDescs();
    if (OutputDescs.Num() != Outputs.Num())
    {
        UE_LOG(LogTemp, Error, TEXT("RunSync failed: Expected %d output tensors, but got %d"), OutputDescs.Num(), Outputs.Num());
        return false;
    }

    UE_LOG(LogTemp, Warning, TEXT("RunSync: Preparing output bindings..."));
    
    TArray<FTensorBindingCPU> OutputBindings;
    OutputBindings.SetNum(Outputs.Num());

    for (int32 i = 0; i < Outputs.Num(); i++)
    {
        if (Outputs[i].Data.Num() == 0)
        {
            UE_LOG(LogTemp, Error, TEXT("RunSync failed: Output tensor at index %d is empty"), i);
            return false;
        }

        OutputBindings[i].Data = (void*)Outputs[i].Data.GetData();
        OutputBindings[i].SizeInBytes = Outputs[i].Data.Num() * sizeof(float);

        UE_LOG(LogTemp, Warning, TEXT("RunSync: Output Tensor %d -> Size: %d bytes"), i, OutputBindings[i].SizeInBytes);
    }

    UE_LOG(LogTemp, Warning, TEXT("RunSync: Running model synchronously..."));

    EResultStatus RunStatus = ModelInstance->RunSync(InputBindings, OutputBindings);
    if (RunStatus != UE::NNE::EResultStatus::Ok)
    {
        UE_LOG(LogTemp, Error, TEXT("RunSync failed: Model execution returned an error"));
        return false;
    }

    UE_LOG(LogTemp, Warning, TEXT("RunSync: Model execution successful!"));
    return true;
}



















TArray<uint8> UNeuralNetworkModel::ProcessScreenshot()
{
    FString ScreenshotFullPath = FPaths::ProjectSavedDir() / TEXT("Screenshots/MacEditor/HighresScreenshot00001.png");
    UE_LOG(LogTemp, Warning, TEXT("Updated Screenshot Path: %s"), *ScreenshotFullPath);

    FString AbsolutePath = FPaths::ConvertRelativePathToFull(ScreenshotFullPath);
    UE_LOG(LogTemp, Warning, TEXT("ProcessScreenshot called with path: %s"), *AbsolutePath);

    if (!FPaths::FileExists(AbsolutePath))
    {
        UE_LOG(LogTemp, Error, TEXT("ProcessScreenshot: File does not exist at path: %s"), *AbsolutePath);
        return {};
    }

    int32 Width, Height;
    TArray<uint8> PixelData = LoadPNGToPixelArray(AbsolutePath, Width, Height);

    if (PixelData.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("ProcessScreenshot: Failed to process screenshot at path: %s"), *AbsolutePath);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("ProcessScreenshot: Successfully loaded screenshot with size: %dx%d"), Width, Height);
    }
    
    // ✅ Print RGB values of the first 5 pixels
        int32 NumPixelsToPrint = 5;
        for (int32 i = 0; i < NumPixelsToPrint; i++)
        {
            int32 Index = i * 4; // Each pixel has 4 bytes (RGBA)
            if (Index + 3 < PixelData.Num())
            {
                uint8 R = PixelData[Index];
                uint8 G = PixelData[Index + 1];
                uint8 B = PixelData[Index + 2];
                uint8 A = PixelData[Index + 3];

                UE_LOG(LogTemp, Warning, TEXT("Pixel %d - R: %d, G: %d, B: %d, A: %d"), i, R, G, B, A);
            }
        }

    return PixelData;
}


TArray<uint8> UNeuralNetworkModel::LoadPNGToPixelArray(const FString& FilePath, int32& OutWidth, int32& OutHeight)
{
    UE_LOG(LogTemp, Warning, TEXT("LoadPNGToPixelArray called with path: %s"), *FilePath);

    // Check if file exists before trying to load it
    if (!FPaths::FileExists(FilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("LoadPNGToPixelArray: File does not exist at path: %s"), *FilePath);
        return {};
    }

    TArray<uint8> RawFileData;
    if (!FFileHelper::LoadFileToArray(RawFileData, *FilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("LoadPNGToPixelArray: Failed to load PNG file: %s"), *FilePath);
        return {};
    }

    UE_LOG(LogTemp, Warning, TEXT("LoadPNGToPixelArray: Successfully loaded file. Data size: %d bytes"), RawFileData.Num());

    // Load the image wrapper module
    IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
    TSharedPtr<IImageWrapper> ImageWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);

    if (!ImageWrapper.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("LoadPNGToPixelArray: Failed to create ImageWrapper"));
        return {};
    }

    if (!ImageWrapper->SetCompressed(RawFileData.GetData(), RawFileData.Num()))
    {
        UE_LOG(LogTemp, Error, TEXT("LoadPNGToPixelArray: Failed to set compressed data for PNG file: %s"), *FilePath);
        return {};
    }

    OutWidth = ImageWrapper->GetWidth();
    OutHeight = ImageWrapper->GetHeight();

    UE_LOG(LogTemp, Warning, TEXT("LoadPNGToPixelArray: Image dimensions: %dx%d"), OutWidth, OutHeight);

    // Extract raw pixel data
    TArray<uint8> RawData;
    if (!ImageWrapper->GetRaw(ERGBFormat::RGBA, 8, RawData))
    {
        UE_LOG(LogTemp, Error, TEXT("LoadPNGToPixelArray: Failed to extract raw pixel data from: %s"), *FilePath);
        return {};
    }

    UE_LOG(LogTemp, Warning, TEXT("LoadPNGToPixelArray: Successfully extracted %d bytes of pixel data"), RawData.Num());
    
    return RawData;
}





















/*
 - Parameters:
    1) RenderTarget: A UTextureRenderTarget2D* representing the render target to process.
 - What it does: Reads pixel data from the render target, normalizes it, and converts it into a tensor format suitable for the model.
 - Return Value: TArray<float> containing the normalized tensor data.
 */

/*
TArray<float> UNeuralNetworkModel::ProcessRenderTarget(UTextureRenderTarget2D* RenderTarget)
{
    if (!RenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("ProcessRenderTarget: RenderTarget is null!"));
        return {};
    }

    FTextureRenderTargetResource* RenderTargetResource = RenderTarget->GameThread_GetRenderTargetResource();
    if (!RenderTargetResource)
    {
        UE_LOG(LogTemp, Error, TEXT("ProcessRenderTarget: Failed to get render target resource!"));
        return {};
    }

    // Get Render Target dimensions
    int32 Width = RenderTarget->SizeX;
    int32 Height = RenderTarget->SizeY;
    UE_LOG(LogTemp, Log, TEXT("ProcessRenderTarget: Render Target Size - Width: %d, Height: %d"), Width, Height);

    // Read pixels from render target
    TArray<FColor> PixelData;
    bool bSuccess = RenderTargetResource->ReadPixels(PixelData);
    if (!bSuccess || PixelData.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("ProcessRenderTarget: Failed to read pixel data!"));
        return {};
    }

    UE_LOG(LogTemp, Log, TEXT("ProcessRenderTarget: Successfully read %d pixels"), PixelData.Num());

    // Convert pixels to a normalized tensor format
    TArray<float> ImageTensor;
    ImageTensor.Reserve(Width * Height * 3);

    for (int32 i = 0; i < PixelData.Num(); i++)
    {
        const FColor& Pixel = PixelData[i];

        // Normalize RGB values to 0-1
        ImageTensor.Add(Pixel.R / 255.0f);
        ImageTensor.Add(Pixel.G / 255.0f);
        ImageTensor.Add(Pixel.B / 255.0f);

        // Log the first 5 pixels for debugging
        if (i < 5)
        {
            UE_LOG(LogTemp, Log, TEXT("Pixel %d: R=%d, G=%d, B=%d -> Normalized: (%.3f, %.3f, %.3f)"),
                i, Pixel.R, Pixel.G, Pixel.B,
                Pixel.R / 255.0f, Pixel.G / 255.0f, Pixel.B / 255.0f);
        }
    }

    UE_LOG(LogTemp, Log, TEXT("ProcessRenderTarget: Successfully converted image to tensor format (Size: %d)"), ImageTensor.Num());

    for(int i = 0; i<ImageTensor.Num();i++) { globalPixelData[i] = ImageTensor[i]; }
    
    return ImageTensor;
}

*/
