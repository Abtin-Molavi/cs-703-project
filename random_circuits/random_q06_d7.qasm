OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
cx q[2],q[5];
cx q[3],q[0];
cx q[1],q[4];
cx q[2],q[3];
cx q[4],q[0];
cx q[5],q[1];
cx q[5],q[3];
cx q[2],q[0];
rz(2.3210511) q[4];
rz(6.2139821) q[1];
rz(3.228451) q[1];
rz(2.6874487) q[3];
cx q[0],q[5];
rz(4.3127808) q[4];
rz(4.3140299) q[2];
rz(4.6125956) q[3];
rz(4.3181082) q[2];
cx q[5],q[4];
rz(2.510059) q[1];
rz(1.112805) q[0];
rz(4.299956) q[4];
cx q[3],q[0];
rz(2.0749502) q[2];
rz(5.098) q[1];
rz(1.6289053) q[5];
cx q[5],q[4];
rz(4.8480042) q[0];
cx q[1],q[2];
rz(2.8863503) q[3];
