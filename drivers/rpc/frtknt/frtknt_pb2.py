# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: frtknt.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='frtknt.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0c\x66rtknt.proto\"\x07\n\x05\x45mpty\"\x15\n\x05MatKW\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"$\n\x04Pair\x12\r\n\x05\x64\x61ta0\x18\x01 \x01(\x05\x12\r\n\x05\x64\x61ta1\x18\x02 \x01(\x05\"\x1c\n\nPointCloud\x12\x0e\n\x06points\x18\x01 \x01(\x0c\"G\n\x06\x43\x61mImg\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\x0f\n\x07\x63hannel\x18\x03 \x01(\x05\x12\r\n\x05image\x18\x04 \x01(\x0c\"T\n\x0ePartialPcdPara\x12\x15\n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32\x07.CamImg\x12\x14\n\x05width\x18\x02 \x01(\x0b\x32\x05.Pair\x12\x15\n\x06height\x18\x03 \x01(\x0b\x32\x05.Pair2\xcd\x01\n\x03Knt\x12\x1e\n\tgetrgbimg\x12\x06.Empty\x1a\x07.CamImg\"\x00\x12 \n\x0bgetdepthimg\x12\x06.Empty\x1a\x07.CamImg\"\x00\x12\x1f\n\x06getpcd\x12\x06.Empty\x1a\x0b.PointCloud\"\x00\x12/\n\rgetpartialpcd\x12\x0f.PartialPcdPara\x1a\x0b.PointCloud\"\x00\x12\x32\n\x1amapColorPointToCameraSpace\x12\x05.Pair\x1a\x0b.PointCloud\"\x00\x62\x06proto3')
)




_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=16,
  serialized_end=23,
)


_MATKW = _descriptor.Descriptor(
  name='MatKW',
  full_name='MatKW',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='MatKW.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=25,
  serialized_end=46,
)


_PAIR = _descriptor.Descriptor(
  name='Pair',
  full_name='Pair',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data0', full_name='Pair.data0', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data1', full_name='Pair.data1', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=48,
  serialized_end=84,
)


_POINTCLOUD = _descriptor.Descriptor(
  name='PointCloud',
  full_name='PointCloud',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='points', full_name='PointCloud.points', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=86,
  serialized_end=114,
)


_CAMIMG = _descriptor.Descriptor(
  name='CamImg',
  full_name='CamImg',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='CamImg.width', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='CamImg.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channel', full_name='CamImg.channel', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image', full_name='CamImg.image', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=116,
  serialized_end=187,
)


_PARTIALPCDPARA = _descriptor.Descriptor(
  name='PartialPcdPara',
  full_name='PartialPcdPara',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='PartialPcdPara.data', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='PartialPcdPara.width', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='PartialPcdPara.height', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=189,
  serialized_end=273,
)

_PARTIALPCDPARA.fields_by_name['data'].message_type = _CAMIMG
_PARTIALPCDPARA.fields_by_name['width'].message_type = _PAIR
_PARTIALPCDPARA.fields_by_name['height'].message_type = _PAIR
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['MatKW'] = _MATKW
DESCRIPTOR.message_types_by_name['Pair'] = _PAIR
DESCRIPTOR.message_types_by_name['PointCloud'] = _POINTCLOUD
DESCRIPTOR.message_types_by_name['CamImg'] = _CAMIMG
DESCRIPTOR.message_types_by_name['PartialPcdPara'] = _PARTIALPCDPARA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), dict(
  DESCRIPTOR = _EMPTY,
  __module__ = 'frtknt_pb2'
  # @@protoc_insertion_point(class_scope:Empty)
  ))
_sym_db.RegisterMessage(Empty)

MatKW = _reflection.GeneratedProtocolMessageType('MatKW', (_message.Message,), dict(
  DESCRIPTOR = _MATKW,
  __module__ = 'frtknt_pb2'
  # @@protoc_insertion_point(class_scope:MatKW)
  ))
_sym_db.RegisterMessage(MatKW)

Pair = _reflection.GeneratedProtocolMessageType('Pair', (_message.Message,), dict(
  DESCRIPTOR = _PAIR,
  __module__ = 'frtknt_pb2'
  # @@protoc_insertion_point(class_scope:Pair)
  ))
_sym_db.RegisterMessage(Pair)

PointCloud = _reflection.GeneratedProtocolMessageType('PointCloud', (_message.Message,), dict(
  DESCRIPTOR = _POINTCLOUD,
  __module__ = 'frtknt_pb2'
  # @@protoc_insertion_point(class_scope:PointCloud)
  ))
_sym_db.RegisterMessage(PointCloud)

CamImg = _reflection.GeneratedProtocolMessageType('CamImg', (_message.Message,), dict(
  DESCRIPTOR = _CAMIMG,
  __module__ = 'frtknt_pb2'
  # @@protoc_insertion_point(class_scope:CamImg)
  ))
_sym_db.RegisterMessage(CamImg)

PartialPcdPara = _reflection.GeneratedProtocolMessageType('PartialPcdPara', (_message.Message,), dict(
  DESCRIPTOR = _PARTIALPCDPARA,
  __module__ = 'frtknt_pb2'
  # @@protoc_insertion_point(class_scope:PartialPcdPara)
  ))
_sym_db.RegisterMessage(PartialPcdPara)



_KNT = _descriptor.ServiceDescriptor(
  name='Knt',
  full_name='Knt',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=276,
  serialized_end=481,
  methods=[
  _descriptor.MethodDescriptor(
    name='getrgbimg',
    full_name='Knt.getrgbimg',
    index=0,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_CAMIMG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getdepthimg',
    full_name='Knt.getdepthimg',
    index=1,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_CAMIMG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getpcd',
    full_name='Knt.getpcd',
    index=2,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_POINTCLOUD,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getpartialpcd',
    full_name='Knt.getpartialpcd',
    index=3,
    containing_service=None,
    input_type=_PARTIALPCDPARA,
    output_type=_POINTCLOUD,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='mapColorPointToCameraSpace',
    full_name='Knt.mapColorPointToCameraSpace',
    index=4,
    containing_service=None,
    input_type=_PAIR,
    output_type=_POINTCLOUD,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_KNT)

DESCRIPTOR.services_by_name['Knt'] = _KNT

# @@protoc_insertion_point(module_scope)