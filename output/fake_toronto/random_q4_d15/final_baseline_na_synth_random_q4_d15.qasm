OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(4.4563702) q[22];
rz(5.1158666) q[24];
rz(1.3304094) q[24];
rz(4.588732) q[24];
cx q[24],q[25];
rz(1.2437687) q[26];
rz(3.0782302) q[26];
rz(0.39031327) q[26];
rz(2.9542407) q[26];
cx q[25],q[26];
rz(2.3516125) q[26];
rz(4.5111409) q[26];
rz(5.9543017) q[26];
rz(4.4881676) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[22],q[25];
rz(0.69329505) q[25];
rz(2.155319) q[25];
rz(5.672957) q[25];
rz(5.0099267) q[25];
cx q[26],q[25];
cx q[25],q[24];
rz(2.2935615) q[24];
rz(0.2847634) q[24];
rz(3.2007435) q[24];
cx q[25],q[24];
rz(3.5103501) q[25];
rz(2.0455454) q[25];
rz(2.8590077) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[26],q[25];
rz(0.2707942) q[26];
rz(0.077874835) q[26];
rz(4.7833148) q[26];
rz(3.4685371) q[26];
