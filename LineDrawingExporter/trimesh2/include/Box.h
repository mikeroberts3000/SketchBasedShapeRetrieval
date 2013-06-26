#ifndef BOX_H
#define BOX_H
/*
Szymon Rusinkiewicz
Princeton University

Box.h
Templated axis-aligned bounding boxes - meant to be used with Vec.h
*/

#include "Vec.h"
#include <iostream>
#include <fstream>
#include <string>


template <int D, class T = float>
class Box {
private:
	typedef Vec<D,T> Point;

public:
	Point min, max;
	bool valid;

	// Construct as empty
	Box() : valid(false)
		{}

	// Mark invalid
	void clear()
		{ valid = false; }

	// Return center point and (vector) diagonal
	Point center() const { return 0.5f * (min+max); }
	Point size() const { return max - min; }

	// Grow a bounding box to encompass a point
	Box<D,T> &operator += (const Point &p)
	{
		if (valid) {
			min.min(p);
			max.max(p);
		} else {
			min = p;
			max = p;
			valid = true;
		}
		return *this;
	}
	Box<D,T> &operator += (const Box<D,T> &b)
	{
		if (valid) {
			min.min(b.min);
			max.max(b.max);
		} else {
			min = b.min;
			max = b.max;
			valid = true;
		}
		return *this;
	}

	friend const Box<D,T> operator + (const Box<D,T> &b, const Point &p)
		{ return Box<D,T>(b) += p; }
	friend const Box<D,T> operator + (const Point &p, const Box<D,T> &b)
		{ return Box<D,T>(b) += p; }
	friend const Box<D,T> operator + (const Box<D,T> &b1, const Box<D,T> &b2)
		{ return Box<D,T>(b1) += b2; }

	// Read a Box from a file.
	bool read(const std::string &filename)
	{
		std::ifstream f(filename.c_str());
		Box<D,T> B;
		f >> B;
		f.close();
		if (f.good()) {
			*this = B;
			return true;
		}
		return false;
	}

	// Write a Box to a file
	bool write(const std::string &filename) const
	{
		std::ofstream f(filename.c_str());
		f << *this;
		f.close();
		return f.good();
	}

	// iostream operators
	friend std::ostream &operator << (std::ostream &os, const Box<D,T> &b)
	{
		const int n = b.min.size();
		for (int i = 0; i < n-1; i++)
			os << b.min[i] << " ";
		os << b.min[n-1] << std::endl;
		for (int i = 0; i < n-1; i++)
			os << b.max[i] << " ";
		os << b.max[n-1] << std::endl;
		return os;
	}
	friend std::istream &operator >> (std::istream &is, Box<D,T> &b)
	{
		const int n = b.min.size();
		for (int i = 0; i < n; i++)
			is >> b.min[i];
		for (int i = 0; i < n; i++)
			is >> b.max[i];
		if (b.min[0] > b.max[0]) swap(b.min[0], b.max[0]);
		if (b.min[1] > b.max[1]) swap(b.min[1], b.max[1]);
		if (b.min[2] > b.max[2]) swap(b.min[2], b.max[2]);
		b.valid = is.good();
		return is;
	}
};

typedef Box<3,float> box;

#endif
