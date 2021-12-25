function xN = Normalize_Fcn(x,MinX,MaxX)

xN = (x - MinX) / (MaxX - MinX) * 2 - 1;

end