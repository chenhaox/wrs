# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import wrs.visualization.panda.rpc.rviz_pb2 as rviz__pb2


class RVizStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.run_code = channel.unary_unary(
                '/RViz/run_code',
                request_serializer=rviz__pb2.CodeRequest.SerializeToString,
                response_deserializer=rviz__pb2.Status.FromString,
                )
        self.create_instance = channel.unary_unary(
                '/RViz/create_instance',
                request_serializer=rviz__pb2.CreateInstanceRequest.SerializeToString,
                response_deserializer=rviz__pb2.Status.FromString,
                )


class RVizServicer(object):
    """Missing associated documentation comment in .proto file."""

    def run_code(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def create_instance(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_RVizServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'run_code': grpc.unary_unary_rpc_method_handler(
                    servicer.run_code,
                    request_deserializer=rviz__pb2.CodeRequest.FromString,
                    response_serializer=rviz__pb2.Status.SerializeToString,
            ),
            'create_instance': grpc.unary_unary_rpc_method_handler(
                    servicer.create_instance,
                    request_deserializer=rviz__pb2.CreateInstanceRequest.FromString,
                    response_serializer=rviz__pb2.Status.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'RViz', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class RViz(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def run_code(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/RViz/run_code',
            rviz__pb2.CodeRequest.SerializeToString,
            rviz__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def create_instance(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/RViz/create_instance',
            rviz__pb2.CreateInstanceRequest.SerializeToString,
            rviz__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
