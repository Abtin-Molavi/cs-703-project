OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.012966502) q[1];
rz(1.2246104) q[2];
cx q[1],q[2];
rz(1.004302) q[2];
rz(3.8836768) q[2];
rz(2.3729661) q[13];
cx q[14],q[11];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
rz(4.9424544) q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[13],q[14];
rz(5.5849565) q[13];
cx q[8],q[11];
cx q[11],q[14];
rz(4.1773213) q[11];
rz(1.7093287) q[8];
rz(3.9487194) q[8];