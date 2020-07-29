#ifndef PARAMETER_LIST_H
#define PARAMETER_LIST_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "Parser.h"
#include <src/my_p4est_utils.h>

class parameter_list_t
{
  struct dbls_t {
    std::vector<double *>    ptrs;
    std::vector<std::string> keys;
    std::vector<std::string> desc;
  } dbls;
  struct ints_t {
    std::vector<int *>       ptrs;
    std::vector<std::string> keys;
    std::vector<std::string> desc;
  } ints;
  struct blns_t {
    std::vector<bool *>      ptrs;
    std::vector<std::string> keys;
    std::vector<std::string> desc;
  } blns;
  struct strs_t {
    std::vector<std::string *> ptrs;
    std::vector<std::string>   keys;
    std::vector<std::string>   desc;
  } strs;
  struct chrs_t {
    std::vector<char **>     ptrs;
    std::vector<std::string> keys;
    std::vector<std::string> desc;
  } chrs;
  struct bcs_t {
    std::vector<BoundaryConditionType *> ptrs;
    std::vector<std::string> keys;
    std::vector<std::string> desc;
  } bcs;
public:
  parameter_list_t();

  void add(double &parameter, const std::string& key, const std::string& description)
  {
    dbls.ptrs.push_back(&parameter);
    dbls.keys.push_back(key);
    dbls.desc.push_back(description);
  }

  void add(int &parameter, const std::string& key, const std::string& description)
  {
    ints.ptrs.push_back(&parameter);
    ints.keys.push_back(key);
    ints.desc.push_back(description);
  }

  void add(bool &parameter, const std::string& key, const std::string& description)
  {
    blns.ptrs.push_back(&parameter);
    blns.keys.push_back(key);
    blns.desc.push_back(description);
  }

  void add(std::string &parameter, const std::string& key, const std::string& description)
  {
    strs.ptrs.push_back(&parameter);
    strs.keys.push_back(key);
    strs.desc.push_back(description);
  }

  void add(char *&parameter, const std::string& key, const std::string& description)
  {
    chrs.ptrs.push_back(&parameter);
    chrs.keys.push_back(key);
    chrs.desc.push_back(description);
  }

  void add(BoundaryConditionType &parameter, const std::string& key, const std::string& description)
  {
    bcs.ptrs.push_back(&parameter);
    bcs.keys.push_back(key);
    bcs.desc.push_back(description);
  }

  void initialize_parser(cmdParser &cmd)
  {
    for (int i = 0; i < dbls.ptrs.size(); ++i) cmd.add_option(dbls.keys[i], dbls.desc[i]);
    for (int i = 0; i < ints.ptrs.size(); ++i) cmd.add_option(ints.keys[i], ints.desc[i]);
    for (int i = 0; i < blns.ptrs.size(); ++i) cmd.add_option(blns.keys[i], blns.desc[i]);
    for (int i = 0; i < strs.ptrs.size(); ++i) cmd.add_option(strs.keys[i], strs.desc[i]);
    for (int i = 0; i < chrs.ptrs.size(); ++i) cmd.add_option(chrs.keys[i], chrs.desc[i]);
    for (int i = 0; i < bcs .ptrs.size(); ++i) cmd.add_option(bcs .keys[i], bcs .desc[i]);
  }

  void get_all(cmdParser &cmd)
  {
    for (int i = 0; i < dbls.ptrs.size(); ++i) (*dbls.ptrs[i]) = cmd.get(dbls.keys[i], *dbls.ptrs[i]);
    for (int i = 0; i < ints.ptrs.size(); ++i) (*ints.ptrs[i]) = cmd.get(ints.keys[i], *ints.ptrs[i]);
    for (int i = 0; i < blns.ptrs.size(); ++i) (*blns.ptrs[i]) = cmd.get(blns.keys[i], *blns.ptrs[i]);
    for (int i = 0; i < strs.ptrs.size(); ++i) (*strs.ptrs[i]) = cmd.get(strs.keys[i], *strs.ptrs[i]);
    for (int i = 0; i < chrs.ptrs.size(); ++i) (*chrs.ptrs[i]) = cmd.get(chrs.keys[i], *chrs.ptrs[i]);
    for (int i = 0; i < bcs .ptrs.size(); ++i) (*bcs .ptrs[i]) = cmd.get(bcs .keys[i], *bcs .ptrs[i]);
  }

  void print_all()
  {
    for (int i = 0; i < dbls.ptrs.size(); ++i) std::cout << dbls.keys[i] << ": " << (*dbls.ptrs[i]) << std::endl;
    for (int i = 0; i < ints.ptrs.size(); ++i) std::cout << ints.keys[i] << ": " << (*ints.ptrs[i]) << std::endl;
    for (int i = 0; i < blns.ptrs.size(); ++i) std::cout << blns.keys[i] << ": " << (*blns.ptrs[i]) << std::endl;
    for (int i = 0; i < strs.ptrs.size(); ++i) std::cout << strs.keys[i] << ": " << (*strs.ptrs[i]) << std::endl;
    for (int i = 0; i < chrs.ptrs.size(); ++i) std::cout << chrs.keys[i] << ": " << (*chrs.ptrs[i]) << std::endl;
    for (int i = 0; i < bcs .ptrs.size(); ++i) std::cout << bcs .keys[i] << ": " << (*bcs .ptrs[i]) << std::endl;
  }

  void save_all(const char output[])
  {
    std::ofstream out(output);
    if (out.is_open())
    {
      for (int i = 0; i < dbls.ptrs.size(); ++i) out << dbls.keys[i] << ": " << (*dbls.ptrs[i]) << std::endl;
      for (int i = 0; i < ints.ptrs.size(); ++i) out << ints.keys[i] << ": " << (*ints.ptrs[i]) << std::endl;
      for (int i = 0; i < blns.ptrs.size(); ++i) out << blns.keys[i] << ": " << (*blns.ptrs[i]) << std::endl;
      for (int i = 0; i < strs.ptrs.size(); ++i) out << strs.keys[i] << ": " << (*strs.ptrs[i]) << std::endl;
      for (int i = 0; i < chrs.ptrs.size(); ++i) out << chrs.keys[i] << ": " << (*chrs.ptrs[i]) << std::endl;
      for (int i = 0; i < bcs .ptrs.size(); ++i) out << bcs .keys[i] << ": " << (*bcs .ptrs[i]) << std::endl;
      out.close();
    }
    else throw std::invalid_argument("Unable to open file\n");
  }
};

class add_to_list
{
public:
  add_to_list(parameter_list_t &list, double &parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
  add_to_list(parameter_list_t &list, bool &parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
  add_to_list(parameter_list_t &list, int &parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
  add_to_list(parameter_list_t &list, std::string &parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
  add_to_list(parameter_list_t &list, char *&parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
  add_to_list(parameter_list_t &list, BoundaryConditionType &parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
};

#define ICAT(A,B) A ## B
#define CAT(A,B) ICAT(A,B)

#define DEFINE_PARAMETER(list, type, var, value, description) type var = value; add_to_list CAT(adding_,var) (list, var, #var, description)
#define ADD_TO_LIST(list, var, description) static add_to_list CAT(adding_,var) (list, var, #var, description)


/*!
 * \brief Classes param_list_t and param_t provides the user with a fast and convenient way to define and manage parameters,
 * TODO: add description
 */
class param_base_t
{
public:
  param_base_t(const std::string& key, const std::string& description)
    : key(key), description(description) {}
  std::string key;
  std::string description;
  virtual void set_from_cmd(cmdParser &cmd)=0;
  virtual std::string print_value()=0;
};

class param_list_t
{
public:
  std::vector<param_base_t *> list;

  void initialize_parser(cmdParser &cmd)
  {
    for (int i = 0; i < list.size(); ++i)
    {
      cmd.add_option(list[i]->key, list[i]->description);
    }
  }

  void set_from_cmd_all(cmdParser &cmd)
  {
    for (int i = 0; i < list.size(); ++i)
    {
      list[i]->set_from_cmd(cmd);
    }
  }

  void print_all(bool align=false)
  {
    int length = 0;

    if (align)
    {
      for (int i = 0; i < list.size(); ++i)
        if (list[i]->key.size() > length)
          length = list[i]->key.size();


      for (int i = 0; i < list.size(); ++i)
      {
        std::cout << list[i]->key << ":" << std::string(length-list[i]->key.size()+1, ' ') << list[i]->print_value() << "\n";
      }
    }
    else
    {
      for (int i = 0; i < list.size(); ++i)
      {
        std::cout << list[i]->key << ": " << list[i]->print_value() << "\n";
      }
    }
  }

  void save_all(const char output[], bool align=false)
  {
    std::ofstream out(output);
    if (out.is_open())
    {
      int length = 0;

      if (align)
      {
        for (int i = 0; i < list.size(); ++i)
          if (list[i]->key.size() > length)
            length = list[i]->key.size();


        for (int i = 0; i < list.size(); ++i)
        {
          out << "-" << list[i]->key << " " << std::string(length-list[i]->key.size()+1, ' ') << list[i]->print_value() << "\n";
        }
      }
      else
      {
        for (int i = 0; i < list.size(); ++i)
        {
          out << "-" << list[i]->key << " " << list[i]->print_value() << "\n";
        }
      }

      out.close();
    }
    else throw std::invalid_argument("Unable to open file\n");
  }
};

template <typename T>
class param_t : public param_base_t
{
public:
  param_t(T value, const std::string& key, const std::string& description)
    : param_base_t(key, description), val(value) {}

  param_t(T value, param_list_t &list, const std::string& key, const std::string& description)
    : param_base_t(key, description), val(value)
  {
    list.list.push_back(this);
  }

  param_t(param_list_t &list, T value, const std::string& key, const std::string& description)
    : param_base_t(key, description), val(value)
  {
    list.list.push_back(this);
  }

  T val;
  inline T operator()() { return val; }
  inline void set(T value) { this->val = value; }
  inline void get() { return val; }
  void set_from_cmd(cmdParser &cmd) { val = cmd.get(key, val); }
  std::string print_value()
  {
    std::ostringstream result;
    result << val;
    return result.str();
  }
};


#endif // PARAMETER_LIST_H
