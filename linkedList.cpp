#include<iostream>
using namespace std;

template<class T>
class Node
{
    Node* n;
    T value;
    public:
    Node(T v, int p = 0): n(NULL), value(v), pos(p){}
    int pos;
    void connect(Node *next)
    {
        n = next;
    }
    
    Node* getNext()
    {
        return n;
    }
    T& getValue()
    {
        return value;
    }
    void nullify()
    {
    	n = NULL;
	}
    ~Node()
    {
            delete n;
    }
};
template<class T>
class List
{
    mutable Node<T>* curr;
    Node<T>* head;
    int size;
    public:
    List(T val)
    {
        size = 1;
        head = new Node<T>(val, size - 1);
        curr = head;
    }
    List(const List& ref)
    {
    	size = 1;
        head = new Node<T>(ref.head->getValue(), size - 1);
        curr = head;
    	for(int i = 1 ; i < ref.getSize() ; i++)
    	{
    		this->append(ref[i]);
		}
	}
    void append(T val)
    {
    	size++;
        Node<T> *temp = new Node<T>(val, size - 1);
        curr->connect(temp);
        curr = temp;
    }

    int getSize() const
    {
    	return size;
	}
	
    T& operator[](int i) const
	{
        curr = head;
        while(curr->pos != i)
        {
            curr = curr->getNext();
        }
        return curr->getValue();
    }
    ~List()
    {
        delete head;
    }
	void print()
    {
    	for(int i = 0 ; i < this->getSize() ; i++)
    	{
    		cout<<(*this)[i]<<endl;
		}
	}
};

int main()
{
    List<string> a("Mahad");
    a.append("Raheem");
    a.append("Abbas");
    a.append("Ahmed");
	a.print();
}
