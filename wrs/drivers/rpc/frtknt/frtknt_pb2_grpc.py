# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc
from . import frtknt_pb2


class KntStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.getrgbimg = channel.unary_unary(
        '/Knt/getrgbimg',
        request_serializer=frtknt_pb2.Empty.SerializeToString,
        response_deserializer=frtknt_pb2.CamImg.FromString,
        )
    self.getdepthimg = channel.unary_unary(
        '/Knt/getdepthimg',
        request_serializer=frtknt_pb2.Empty.SerializeToString,
        response_deserializer=frtknt_pb2.CamImg.FromString,
        )
    self.getpcd = channel.unary_unary(
        '/Knt/getpcd',
        request_serializer=frtknt_pb2.Empty.SerializeToString,
        response_deserializer=frtknt_pb2.PointCloud.FromString,
        )
    self.getpartialpcd = channel.unary_unary(
        '/Knt/getpartialpcd',
        request_serializer=frtknt_pb2.PartialPcdPara.SerializeToString,
        response_deserializer=frtknt_pb2.PointCloud.FromString,
        )
    self.mapColorPointToCameraSpace = channel.unary_unary(
        '/Knt/mapColorPointToCameraSpace',
        request_serializer=frtknt_pb2.Pair.SerializeToString,
        response_deserializer=frtknt_pb2.PointCloud.FromString,
        )


class KntServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def getrgbimg(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def getdepthimg(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def getpcd(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def getpartialpcd(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def mapColorPointToCameraSpace(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_KntServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'getrgbimg': grpc.unary_unary_rpc_method_handler(
          servicer.getrgbimg,
          request_deserializer=frtknt_pb2.Empty.FromString,
          response_serializer=frtknt_pb2.CamImg.SerializeToString,
      ),
      'getdepthimg': grpc.unary_unary_rpc_method_handler(
          servicer.getdepthimg,
          request_deserializer=frtknt_pb2.Empty.FromString,
          response_serializer=frtknt_pb2.CamImg.SerializeToString,
      ),
      'getpcd': grpc.unary_unary_rpc_method_handler(
          servicer.getpcd,
          request_deserializer=frtknt_pb2.Empty.FromString,
          response_serializer=frtknt_pb2.PointCloud.SerializeToString,
      ),
      'getpartialpcd': grpc.unary_unary_rpc_method_handler(
          servicer.getpartialpcd,
          request_deserializer=frtknt_pb2.PartialPcdPara.FromString,
          response_serializer=frtknt_pb2.PointCloud.SerializeToString,
      ),
      'mapColorPointToCameraSpace': grpc.unary_unary_rpc_method_handler(
          servicer.mapColorPointToCameraSpace,
          request_deserializer=frtknt_pb2.Pair.FromString,
          response_serializer=frtknt_pb2.PointCloud.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Knt', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
