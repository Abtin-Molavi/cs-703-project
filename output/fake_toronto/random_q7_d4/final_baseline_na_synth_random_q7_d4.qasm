OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.3729661) q[3];
cx q[3],q[2];
cx q[8],q[5];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[3],q[2];
rz(1.7093287) q[3];
rz(3.9487194) q[3];
rz(4.9424544) q[11];
cx q[11],q[8];
cx q[8],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[5],q[3];
cx q[2],q[3];
rz(4.1773213) q[2];
rz(5.5849565) q[8];
rz(0.012966502) q[24];
rz(1.2246104) q[25];
cx q[24],q[25];
rz(1.004302) q[25];
rz(3.8836768) q[25];