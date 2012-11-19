#ifndef _CHOHONG_ArrayV_____
#define _CHOHONG_ArrayV_____

#include <assert.h>
#include <memory.h>
#include <stdexcept>
#include <iostream>
#include <stdlib.h>
#include <cstring>
// CASL
#include <lib/utilities/Macros.h>

#define _DEBUG

using namespace std;

//----------------------------------------------------
//
// By Chohong Min, 2005
//
// ArrayV : [0]~[m_size-1]~[m_max-1]
//
//----------------------------------------------------
template< class  T>
class ArrayV
{
protected:
    T*  m_data;
    CaslInt	m_size;
    CaslInt m_max;
public:
    std::string name;

public:

    void reallocate(CaslInt size) {
      if (NULL != m_data) delete [] m_data; // free(m_data);
        m_size = size; m_max = size;
        if (0 != size) m_data = new T[m_max];//(T*)malloc(m_max*sizeof(T));
    }

    void resize_without_copy( CaslInt size )
    {
        if(m_max < size)
        {
            if(m_data!=NULL) {
                delete [] m_data; //  free(m_data);
                m_data=NULL;
            }

            m_max  = 2*size+1;
            m_size =   size;

            if(size !=0) m_data = new T[m_max]; //(T*)malloc(m_max*sizeof(T));
            else         m_data = NULL;
        }
        else
            m_size = size;
    }

    double dot( const ArrayV<T>& V ) const
    {
      double  ret=0;
      const T* data = (const T*)V;
      for( int i=0; i<m_size; i++ )
        ret += m_data[i] * data[i];
      return ret;
    }

    void resize_with_copy( CaslInt size )
    {
        if(m_max < size )
        {
            CaslInt copy_size = (m_size<size) ? m_size : size;
            CaslInt i;
            T* new_data = new T[copy_size]; //(T*)malloc(copy_size*sizeof(T));
            for(i=0;i<copy_size;i++) {
                new_data[i] =   m_data[i];
            }
            resize_without_copy(size);
            for(i=0;i<copy_size;i++)   {
                m_data[i] = new_data[i];
            }
            delete [] new_data; //free(new_data);
        }
        else
            m_size = size;
    }

    ArrayV( CaslInt size = 0 )
    {
        name="ArrayV";
        m_data = NULL;
        m_size = 0;
        m_max  = 0; reallocate(size);
    }

    ~ArrayV()
    {
        m_max = 0;
        m_size =0;
        if(m_data!=NULL) {
            delete []  m_data; //free(m_data);
            m_data = NULL;
        }
    }

    void clear() {
        if (m_data != NULL) {
          delete [] m_data; //free(m_data);
          m_data = NULL;
        }
        m_size = 0;
        m_max  = 0;
    }

    ArrayV(const ArrayV<T>& V )
    {
        m_data = NULL;
        m_size = 0;
        m_max  = 0;

        resize_without_copy(V.m_size);

        for(CaslInt i=0;i<m_size;i++)
            m_data[i]=V.m_data[i];
    }

    bool contains(const T& element){
        for (CaslInt n=0; n<m_size; n++){
            if (m_data[n] == element) return true;
        }
        return false;
    }

    void operator=(const ArrayV<T>& V )
    {
        resize_without_copy(V.m_size);

        for(CaslInt i=0;i<m_size;i++)
            m_data[i]=V.m_data[i];
    }

    void push( const T& element )
    {
        if(m_size==m_max)
        {
            CaslInt copy_size = m_size;

            CaslInt increment      = 2*m_size+1;
            //            CaslInt increment_200M = 200000000/sizeof(T);

            resize_with_copy(m_size+increment);//MIN(increment,increment_100M));
            m_size = copy_size;
        }
        m_data[m_size++]=element;
    }

    T pop( )
    {
        m_size--;
        return m_data[m_size];
    }

    void push_with_maintaining_small_memory( const T& element )
    {
        if(m_size==m_max)
        {
            CaslInt copy_size = m_size;
            resize_with_copy(m_size+1);
            m_size = copy_size;
        }
        m_data[m_size++]=element;
    }

    void insert( CaslInt index_insert, const T& element )
    {
        if(m_size==m_max)
        {
            CaslInt copy_size = m_size;
            resize_with_copy(2*m_size+100);
            m_size = copy_size;
        }

        for(CaslInt i=m_size;i>index_insert;i--)
            m_data[i] = m_data[i-1];
        m_size++;
        m_data[index_insert] = element;
    }

    void remove( CaslInt index_remove )
    {
        for(CaslInt i=index_remove;i<m_size-1;i++) m_data[i] = m_data[i+1];
        m_size--;
    }

    //-----------------------------------------
    // Accessors
    //-----------------------------------------
    CaslInt        size() const{ return m_size;}
    CaslInt buffer_size() const{ return m_max ;}

    inline T& operator()(CaslInt i)
    {
#ifdef CASL_THROWS
        if(i<0 || i>=m_size) throw domain_error("[CASL_ERROR]: Index out of bound.");
#endif

        return m_data[i];
    }

    inline const T&  operator()(CaslInt i) const
    {
#ifdef CASL_THROWS
        if(i<0 || i>=m_size) throw domain_error("[CASL_ERROR]: Index out of bound.");
#endif

        return m_data[i];
    }

    inline const T&  operator[](CaslInt i) const
    {
#ifdef CASL_THROWS
        if(i<0 || i>=m_size) throw domain_error("[CASL_ERROR]: Index out of bound.");
#endif

        return m_data[i];
    }

    double abs_max() const
    {
        double max = 0;
        for(CaslInt i=0;i<m_size;i++){ double s=m_data[i]; s=(s>0)?s:-s; if(s>max)max=s;}
        return max;
    }
    double avg_abs() const
    {
        double sum = 0;
        for(long i=0;i<m_size;i++){ sum+=ABS(m_data[i]);}
        return sum/m_size;
    }
    double sum() const {
        double sum = 0;
        for(long i=0;i<m_size;i++){ sum+=m_data[i];}
        return sum;
    }
    double max() const
    {
        double max = m_data[0];
        for(long i=0;i<m_size;i++){ double s=m_data[i]; if(s>max)max=s;}
        return max;
    }
    double min() const
    {
        double min = m_data[0];
        for(long i=0;i<m_size;i++){ double s=m_data[i];  if(s<min)min=s;}
        return min;
    }

    void abs() const
    {
        for (CaslInt i=0; i<m_size; i++) m_data[i] = ABS(m_data[i]);
    }

    void operator=( T s ){
//#pragma omp  parallel for
        for( CaslInt i=0; i<m_size; i++ ) m_data[i] = s;
    }

    operator const T*() const{ return m_data; }
    operator       T*()      { return m_data; }

    void operator*=(double s){
//#pragma omp parallel for
        for(CaslInt i=0;i<m_size;i++ ) m_data[i]*=s;
    }
    void operator/=(double s){
//#pragma omp parallel for
        for(CaslInt i=0;i<m_size;i++ ) m_data[i]/=s;
    }
    void operator+=(const ArrayV<T>& A){
//#pragma omp parallel for
        for(CaslInt i=0;i<m_size;i++ ) m_data[i]+=A.m_data[i];
    }
    void operator-=(const ArrayV<T>& A){
        //#pragma omp parallel for
        for(CaslInt i=0;i<m_size;i++ )
            m_data[i]-=A.m_data[i];
    }
    void operator+=(double s){
//#pragma omp parallel for
        for(CaslInt i=0;i<m_size;i++ ) m_data[i]+=s;}
    void operator-=(double s){
//#pragma omp parallel for
        for(CaslInt i=0;i<m_size;i++ ) m_data[i]-=s;
    }

    void print( const char* name=0 ) const
    {
        cout.setf(ios::showpos);
        cout.setf(ios::fixed);
        cout.precision(12);
        printf("---------------%s---------------\n",name);
        for( CaslInt i=0; i<m_size; i++ )
            cout << m_data[i] << ' ' ;
        cout << endl;
    }

    //---------------------------------------------------------------------
    // FILE IO
    //---------------------------------------------------------------------
    void save( const char* file_name ) const
    {
        FILE* fp = fopen(file_name,"wb");

        CaslInt data_size = sizeof(T)*m_size;

        fwrite( (void*)&m_size,sizeof(char),sizeof(CaslInt)     ,fp);
        fwrite( (void*) m_data,sizeof(char),data_size,fp);

        fclose(fp);
    }

    void load( const char* file_name )
    {
        FILE* fp = fopen(file_name,"rb");
#ifdef CASL_THROWS
        if (fp == NULL){
            throw runtime_error("[CASL_ERROR]: Could not open file " + string(file_name));
        }
#endif

        CaslInt size;

        fread( (void*)&size  ,sizeof(char),sizeof(CaslInt)     ,fp); resize_without_copy(size);
        CaslInt data_size = sizeof(T)*m_size;
        fread( (void*) m_data,sizeof(char),data_size,fp);

        fclose(fp);
    }

    size_t size_in_bytes(){
        return (sizeof(string::size_type) + name.size() + sizeof(CaslInt) + m_size*sizeof(T));
    }

    void serialize(std::ostream& data){
        data.write((char*)&m_size, sizeof(CaslInt));
        data.write((char*)&m_data[0], sizeof(T)*m_size);
    }

    void deserialize(std::istream& data){
        CaslInt size;
        data.read((char*)&size, sizeof(CaslInt));
        clear();
        resize_without_copy(size);
        data.read((char*)&m_data[0], sizeof(T)*m_size);
    }

    void serialize(char* p_data){
        string::size_type name_size = name.size();
        memcpy((void*)p_data, (const void*)&name_size, sizeof(string::size_type)); p_data += sizeof(string::size_type);
        memcpy((void*)p_data, (const void*)name.c_str(), name.size()); p_data += name.size();
        memcpy((void*)p_data, (void*)&m_size, sizeof(CaslInt)); p_data += sizeof(CaslInt);
        memcpy((void*)p_data, (void*)&m_data[0], sizeof(T)*m_size);
    }

    void deserialize(const char* p_data){
        clear();

        char my_name [1024];
        string::size_type my_name_size;
        memcpy((void*)&my_name_size, (const void*)p_data, sizeof(string::size_type)); p_data += sizeof(string::size_type);
        memcpy((void*)&my_name[0],   (const void*)p_data, my_name_size); p_data += my_name_size;
        name = (const char*)&my_name[0];

        CaslInt size;
        memcpy((void*)&size, (void*)p_data, sizeof(CaslInt)); p_data += sizeof(CaslInt);

        reallocate(size);

        memcpy((void*)&m_data[0], (void*)p_data, sizeof(T)*m_size);
    }


    void save_as_m_file_format(const std::string& name){
        std::string file_name = name + ".m";
        std::string var_name  = name + "_";

        FILE* fp = fopen(file_name.c_str(), "w");
        fprintf(fp,"%s = [",var_name.c_str());
        for(CaslInt i=0; i<m_size; i++){
            fprintf(fp, "%1.16e;\n",m_data[i]);
        }
        fprintf(fp,"];");
        fclose(fp);
    }

    //---------------------------------------------------------------------
    // check for NAN
    //---------------------------------------------------------------------
    void CHK_NAN () const
    {
        for (CaslInt n = 0; n<m_size; n++){
            if (my_isnan(m_data[n])) {printf("Warning NAN detected ... Elem(%3d) = %f\n",n,m_data[n]); getchar();}
        }
    }

    //---------------------------------------------------------------------
    // check for INF
    //---------------------------------------------------------------------
    void CHK_INF() const
    {
        for (CaslInt n = 0; n<m_size; n++){
            if (isinf(m_data[n])) {printf("Warning INF detected ... Elem(%3d) = %f\n",n,m_data[n]); getchar();}
        }
    }

    void setData(T* m_data_, CaslInt m_size_){
        m_data = m_data_;
        m_size = m_size_;
        if (m_max < m_size) m_max = m_size;
    }

    friend ostream& operator << (ostream& os, const ArrayV<T>& data){
        for (CaslInt n=0; n<data.size(); n++)
            os << data.name << "(" << n << ") = " << data(n) << "\n";
        os << endl;

        return os;
    }

};

#endif
