OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.0060558) q[2];
rz(3.8060682) q[2];
rz(0.73468205) q[2];
rz(2.9965474) q[10];
rz(1.9157338) q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
rz(3.3313077) q[13];
cx q[14],q[11];
rz(3.3101563) q[11];
rz(3.1644893) q[11];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[15];
cx q[18],q[21];
cx q[15],q[18];
rz(4.3658624) q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[10],q[12];
rz(2.960214) q[10];
rz(0.26136329) q[21];
rz(0.080164595) q[21];
rz(5.8672628) q[26];
rz(5.2646964) q[26];
rz(3.7951887) q[26];
