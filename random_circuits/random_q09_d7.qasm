OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2],q[6];
cx q[5],q[8];
cx q[3],q[0];
rz(2.1976126) q[4];
cx q[1],q[7];
cx q[4],q[3];
cx q[2],q[0];
rz(6.109207) q[1];
rz(2.383605) q[6];
cx q[7],q[5];
rz(3.5473165) q[8];
rz(1.3895899) q[1];
rz(2.9450287) q[2];
cx q[6],q[4];
rz(0.90194373) q[7];
rz(1.6372937) q[5];
cx q[8],q[0];
rz(1.898658) q[3];
cx q[0],q[8];
cx q[6],q[7];
rz(2.9293817) q[4];
rz(4.5428904) q[2];
rz(5.4168182) q[3];
rz(5.0820965) q[1];
rz(5.5768148) q[5];
cx q[3],q[2];
cx q[6],q[0];
rz(4.9362919) q[5];
rz(0.76485436) q[4];
rz(3.3647955) q[8];
cx q[1],q[7];
cx q[1],q[8];
rz(3.8502912) q[0];
rz(1.4381044) q[2];
cx q[6],q[7];
cx q[5],q[4];
rz(3.1379044) q[3];
rz(5.3218886) q[7];
cx q[6],q[1];
cx q[2],q[4];
cx q[0],q[3];
cx q[8],q[5];
