# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/object_server.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/object_server.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1aprotos/object_server.proto\"1\n\rObjectRequest\x12\n\n\x02ws\x18\x01 \x01(\t\x12\x14\n\x05\x61reas\x18\x02 \x03(\x0b\x32\x05.Area\"K\n\x04\x41rea\x12\x0f\n\x07\x61rea_id\x18\x01 \x01(\t\x12\x1c\n\x0e\x64\x65tection_area\x18\x02 \x01(\x0b\x32\x04.Box\x12\x14\n\x04poly\x18\x03 \x03(\x0b\x32\x06.Point\"\x1d\n\x05Point\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\"U\n\x11\x44\x65tectObjResponse\x12\x18\n\nforgot_obj\x18\x01 \x03(\x0b\x32\x04.Obj\x12\x11\n\ttimestamp\x18\x03 \x01(\x01\x12\x13\n\x0bimage_bytes\x18\x04 \x01(\x0c\"G\n\x03Obj\x12\x12\n\nconfidence\x18\x01 \x01(\x02\x12\x1a\n\x0c\x62ounding_box\x18\x02 \x01(\x0b\x32\x04.Box\x12\x10\n\x08track_id\x18\x03 \x01(\t\":\n\x03\x42ox\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\r\n\x05width\x18\x03 \x01(\x02\x12\x0e\n\x06height\x18\x04 \x01(\x02\x32\xc9\x01\n\rObjectService\x12=\n\x13ViolateObjectDetect\x12\x0e.ObjectRequest\x1a\x12.DetectObjResponse\"\x00\x30\x01\x12;\n\x11LeaveObjectDetect\x12\x0e.ObjectRequest\x1a\x12.DetectObjResponse\"\x00\x30\x01\x12<\n\x12\x46orgotObjectDetect\x12\x0e.ObjectRequest\x1a\x12.DetectObjResponse\"\x00\x30\x01\x62\x06proto3'
)




_OBJECTREQUEST = _descriptor.Descriptor(
  name='ObjectRequest',
  full_name='ObjectRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ws', full_name='ObjectRequest.ws', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='areas', full_name='ObjectRequest.areas', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=30,
  serialized_end=79,
)


_AREA = _descriptor.Descriptor(
  name='Area',
  full_name='Area',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='area_id', full_name='Area.area_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='detection_area', full_name='Area.detection_area', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='poly', full_name='Area.poly', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=81,
  serialized_end=156,
)


_POINT = _descriptor.Descriptor(
  name='Point',
  full_name='Point',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='Point.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='Point.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=158,
  serialized_end=187,
)


_DETECTOBJRESPONSE = _descriptor.Descriptor(
  name='DetectObjResponse',
  full_name='DetectObjResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='forgot_obj', full_name='DetectObjResponse.forgot_obj', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='DetectObjResponse.timestamp', index=1,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image_bytes', full_name='DetectObjResponse.image_bytes', index=2,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_end=274,
)


_OBJ = _descriptor.Descriptor(
  name='Obj',
  full_name='Obj',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='confidence', full_name='Obj.confidence', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bounding_box', full_name='Obj.bounding_box', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='track_id', full_name='Obj.track_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=276,
  serialized_end=347,
)


_BOX = _descriptor.Descriptor(
  name='Box',
  full_name='Box',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='Box.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='Box.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width', full_name='Box.width', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='height', full_name='Box.height', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=349,
  serialized_end=407,
)

_OBJECTREQUEST.fields_by_name['areas'].message_type = _AREA
_AREA.fields_by_name['detection_area'].message_type = _BOX
_AREA.fields_by_name['poly'].message_type = _POINT
_DETECTOBJRESPONSE.fields_by_name['forgot_obj'].message_type = _OBJ
_OBJ.fields_by_name['bounding_box'].message_type = _BOX
DESCRIPTOR.message_types_by_name['ObjectRequest'] = _OBJECTREQUEST
DESCRIPTOR.message_types_by_name['Area'] = _AREA
DESCRIPTOR.message_types_by_name['Point'] = _POINT
DESCRIPTOR.message_types_by_name['DetectObjResponse'] = _DETECTOBJRESPONSE
DESCRIPTOR.message_types_by_name['Obj'] = _OBJ
DESCRIPTOR.message_types_by_name['Box'] = _BOX
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ObjectRequest = _reflection.GeneratedProtocolMessageType('ObjectRequest', (_message.Message,), {
  'DESCRIPTOR' : _OBJECTREQUEST,
  '__module__' : 'protos.object_server_pb2'
  # @@protoc_insertion_point(class_scope:ObjectRequest)
  })
_sym_db.RegisterMessage(ObjectRequest)

Area = _reflection.GeneratedProtocolMessageType('Area', (_message.Message,), {
  'DESCRIPTOR' : _AREA,
  '__module__' : 'protos.object_server_pb2'
  # @@protoc_insertion_point(class_scope:Area)
  })
_sym_db.RegisterMessage(Area)

Point = _reflection.GeneratedProtocolMessageType('Point', (_message.Message,), {
  'DESCRIPTOR' : _POINT,
  '__module__' : 'protos.object_server_pb2'
  # @@protoc_insertion_point(class_scope:Point)
  })
_sym_db.RegisterMessage(Point)

DetectObjResponse = _reflection.GeneratedProtocolMessageType('DetectObjResponse', (_message.Message,), {
  'DESCRIPTOR' : _DETECTOBJRESPONSE,
  '__module__' : 'protos.object_server_pb2'
  # @@protoc_insertion_point(class_scope:DetectObjResponse)
  })
_sym_db.RegisterMessage(DetectObjResponse)

Obj = _reflection.GeneratedProtocolMessageType('Obj', (_message.Message,), {
  'DESCRIPTOR' : _OBJ,
  '__module__' : 'protos.object_server_pb2'
  # @@protoc_insertion_point(class_scope:Obj)
  })
_sym_db.RegisterMessage(Obj)

Box = _reflection.GeneratedProtocolMessageType('Box', (_message.Message,), {
  'DESCRIPTOR' : _BOX,
  '__module__' : 'protos.object_server_pb2'
  # @@protoc_insertion_point(class_scope:Box)
  })
_sym_db.RegisterMessage(Box)



_OBJECTSERVICE = _descriptor.ServiceDescriptor(
  name='ObjectService',
  full_name='ObjectService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=410,
  serialized_end=611,
  methods=[
  _descriptor.MethodDescriptor(
    name='ViolateObjectDetect',
    full_name='ObjectService.ViolateObjectDetect',
    index=0,
    containing_service=None,
    input_type=_OBJECTREQUEST,
    output_type=_DETECTOBJRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='LeaveObjectDetect',
    full_name='ObjectService.LeaveObjectDetect',
    index=1,
    containing_service=None,
    input_type=_OBJECTREQUEST,
    output_type=_DETECTOBJRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ForgotObjectDetect',
    full_name='ObjectService.ForgotObjectDetect',
    index=2,
    containing_service=None,
    input_type=_OBJECTREQUEST,
    output_type=_DETECTOBJRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_OBJECTSERVICE)

DESCRIPTOR.services_by_name['ObjectService'] = _OBJECTSERVICE

# @@protoc_insertion_point(module_scope)
