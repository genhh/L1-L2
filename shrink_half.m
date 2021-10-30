

function z = shrink_half(x, lamda,u)

    phi = acos(lamda/8.*(abs(x)/3).^(-1.5)); %»¡¶È
    z = 2/3.*x.*(1+cos(2*pi/3-2/3*phi));
    
    z(find(abs(x) <= (54)^(1/3)/4*(lamda*u)^(2/3))) = 0;


end
