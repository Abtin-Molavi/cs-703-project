OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(0.80342333) q[12];
cx q[15],q[5];
cx q[13],q[3];
rz(3.9018641) q[9];
cx q[10],q[4];
rz(2.9227544) q[1];
cx q[6],q[7];
rz(1.3507266) q[8];
cx q[14],q[11];
rz(0.73549767) q[0];
rz(1.1130027) q[2];
rz(2.950764) q[8];
rz(2.7380787) q[1];
cx q[6],q[3];
rz(1.7420319) q[10];
cx q[4],q[13];
rz(4.3482583) q[14];
cx q[11],q[5];
rz(3.7572149) q[0];
cx q[9],q[12];
cx q[7],q[15];
rz(4.7287338) q[2];
cx q[3],q[10];
cx q[15],q[4];
rz(5.5304048) q[7];
rz(5.3831131) q[6];
cx q[11],q[9];
rz(5.7760613) q[8];
rz(3.9859953) q[1];
cx q[13],q[12];
cx q[2],q[14];
cx q[0],q[5];
cx q[2],q[15];
cx q[8],q[10];
cx q[1],q[3];
cx q[6],q[14];
rz(5.3143373) q[5];
rz(2.0863753) q[13];
cx q[9],q[0];
rz(4.5779556) q[12];
cx q[11],q[7];
rz(5.1716912) q[4];