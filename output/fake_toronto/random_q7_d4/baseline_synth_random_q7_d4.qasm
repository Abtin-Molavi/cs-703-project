OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.012966502) q[0];
rz(1.2246104) q[1];
rz(4.9424544) q[3];
rz(2.3729661) q[4];
cx q[6],q[5];
cx q[4],q[2];
cx q[3],q[6];
cx q[6],q[4];
cx q[5],q[2];
cx q[0],q[1];
cx q[2],q[4];
rz(1.004302) q[1];
rz(3.8836768) q[1];
rz(4.1773213) q[2];
rz(1.7093287) q[5];
rz(3.9487194) q[5];
rz(5.5849565) q[6];
