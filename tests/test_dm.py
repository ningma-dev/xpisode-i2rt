from xpisode_i2rt.arm import KINDS
from xpisode_i2rt.dm import decode, encode


def test_encodes_what_the_rust_side_encodes() -> None:
    # src/dm.rs encodes_what_i2rt_encodes; i2rt's own float_to_uint gave these integers.
    assert encode(KINDS["DM4340"], 1.0, -2.0, 80.0, 5.0, 3.0) == bytes(
        [0x8A, 0x3C, 0x66, 0x62, 0x8F, 0xFF, 0xF8, 0xDA]
    )


def test_decodes_a_reply() -> None:
    reply = decode(KINDS["DM4310"], bytes([0x11, 0x80, 0x00, 0x80, 0x08, 0x00, 30, 31]))
    assert reply.state == 1
    assert abs(reply.position) < 0.001
    assert abs(reply.velocity) < 0.02
    assert abs(reply.torque) < 0.01
