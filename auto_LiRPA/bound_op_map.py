from auto_LiRPA.bound_ops import *

bound_op_map = {
    'onnx::Conv': BoundConv2d,
    'onnx::BatchNormalization': BoundBatchNorm2d,
    'onnx::Relu': BoundReLU,
    'onnx::Tanh': BoundTanh,
    'onnx::Sigmoid': BoundSigmoid,
    'onnx::Exp': BoundExp,
    # 'onnx::MaxPool': BoundMaxPool2d,
    'onnx::Add': BoundAdd,
    'onnx::Sub': BoundSub,
    'onnx::Mul': BoundMul,
    'onnx::Div': BoundDiv,
    'onnx::Neg': BoundNeg,
    # 'onnx::GlobalAveragePool': AdaptiveAvgPool2d,
    'onnx::AveragePool': BoundAvgPool2d,
    'onnx::Reshape': BoundReshape,
    'onnx::Concat': BoundConcat,
    'onnx::Pad': BoundPad,
    'onnx::Gemm': BoundLinear,
    'onnx::Unsqueeze': BoundUnsqueeze,
    'onnx::Squeeze': BoundSqueeze,
    'onnx::ConstantOfShape': BoundConstantOfShape,
    'onnx::Constant': BoundConstant,
    'onnx::Shape': BoundShape,
    'onnx::Gather': BoundGather,
    'aten::gather': BoundGatherAten,
    'onnx::GatherElements': BoundGatherElements,
    'prim::Constant': BoundPrimConstant,
    'onnx::RNN': BoundRNN,
    'onnx::Transpose': BoundTranspose,
    'onnx::MatMul': BoundMatMul,
    'onnx::Cast': BoundCast,
    'onnx::Softmax': BoundSoftmax,
    'onnx::ReduceMean': BoundReduceMean,
    'onnx::ReduceSum': BoundReduceSum,
    'onnx::Dropout': BoundDropout,
    'onnx::Split': BoundSplit,
    'onnx::LeakyRelu': BoundLeakyRelu
}