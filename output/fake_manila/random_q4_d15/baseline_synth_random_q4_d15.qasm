OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(1.2437687) q[2];
rz(3.0782302) q[2];
rz(0.39031327) q[2];
rz(2.9542407) q[2];
rz(5.1158666) q[0];
rz(1.3304094) q[0];
rz(4.588732) q[0];
rz(4.4563702) q[3];
cx q[0],q[1];
cx q[1],q[2];
rz(2.3516125) q[2];
rz(4.5111409) q[2];
rz(5.9543017) q[2];
rz(4.4881676) q[2];
cx q[3],q[2];
rz(0.69329505) q[2];
rz(2.155319) q[2];
rz(5.672957) q[2];
rz(5.0099267) q[2];
cx q[1],q[2];
cx q[2],q[0];
rz(2.2935615) q[0];
rz(0.2847634) q[0];
rz(3.2007435) q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(3.5103501) q[2];
rz(2.0455454) q[2];
rz(0.2707942) q[1];
rz(0.077874835) q[1];
rz(4.7833148) q[1];
rz(3.4685371) q[1];
rz(2.8590077) q[2];
